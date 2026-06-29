import asyncio
import aiohttp
from bs4 import BeautifulSoup
import urllib.robotparser
from urllib.parse import urlparse, urljoin
import json
import os
import argparse
import random
import time
import re
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crawler.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("FinnyCrawler")

# List of realistic user agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edge/120.0.0.0"
]

class RobotsCache:
    """Caches robots.txt rules for domains to avoid re-fetching them frequently."""
    def __init__(self):
        self.parsers = {}

    async def can_fetch(self, url, user_agent="*"):
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = urljoin(base_url, "/robots.txt")

        if base_url not in self.parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                # Read robots.txt asynchronously in an executor to prevent blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, rp.read)
                self.parsers[base_url] = rp
            except Exception as e:
                # If robots.txt cannot be fetched or parsed, assume default allow
                logger.warning(f"Could not read robots.txt from {robots_url}: {e}")
                self.parsers[base_url] = None

        rp = self.parsers[base_url]
        if rp is None:
            return True
        return rp.can_fetch(user_agent, url)


class FinnyCrawler:
    def __init__(self, seed_urls, output_file, max_pages=1000, max_concurrency=10, delay=1.0):
        self.seed_urls = seed_urls
        self.output_file = output_file
        self.max_pages = max_pages
        self.max_concurrency = max_concurrency
        self.delay = delay
        
        self.queue = asyncio.Queue()
        self.visited = set()
        self.robots_cache = RobotsCache()
        self.pages_scraped = 0
        
        # Keep track of active domains to apply politeness delays
        self.last_request_time = {}
        # Keep track of blocked/failed domains for backoff
        self.domain_cooldowns = {}
        
        # Write lock for jsonl output file
        self.write_lock = asyncio.Lock()

        # Initialize queue with seeds
        for url in self.seed_urls:
            self.queue.put_nowait(url)

    def clean_text(self, html_content):
        """Extracts clean text from HTML, removing scripts, styles, and unwanted tags."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator=" ")
        
        # Break into lines and remove leading/trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        # Get title
        title = soup.title.string.strip() if soup.title else ""
        
        return title, clean_text

    def extract_links(self, html_content, base_url):
        """Extracts valid URLs from the page for further crawling."""
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor['href']
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            # Only follow http/https links and discard fragment identifiers
            if parsed.scheme in ("http", "https"):
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.append(clean_url)
        return links

    async def fetch_page(self, session, url):
        """Fetches the page HTML content while respecting politeness and cooldown rules."""
        parsed = urlparse(url)
        domain = parsed.netloc

        # Check domain cooldowns
        if domain in self.domain_cooldowns:
            cooldown_until = self.domain_cooldowns[domain]
            if time.time() < cooldown_until:
                # Re-queue the URL for later crawling and return None
                return None

        # Check robots.txt permissions
        if not await self.robots_cache.can_fetch(url):
            logger.info(f"Ignored (blocked by robots.txt): {url}")
            return None

        # Politeness delay per domain
        last_time = self.last_request_time.get(domain, 0)
        elapsed = time.time() - last_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)

        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }

        try:
            self.last_request_time[domain] = time.time()
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    # Successful fetch
                    # Remove cooldown if it was in cooldown before
                    self.domain_cooldowns.pop(domain, None)
                    html = await response.text()
                    return html
                elif response.status == 429:
                    logger.warning(f"Rate limited (429) by {domain}. Backing off.")
                    self.domain_cooldowns[domain] = time.time() + 60.0  # 60s cooldown
                else:
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
            # Temporary domain cooldown on connection failure
            self.domain_cooldowns[domain] = time.time() + 10.0

        return None

    async def save_to_jsonl(self, data):
        """Thread-safe writing of scraped page data to JSONL."""
        async with self.write_lock:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def worker(self, session):
        """Worker loop that processes URLs from the queue."""
        while self.pages_scraped < self.max_pages:
            try:
                url = await self.queue.get()
            except asyncio.QueueEmpty:
                break

            if url in self.visited:
                self.queue.task_done()
                continue

            self.visited.add(url)
            logger.info(f"Crawling ({self.pages_scraped}/{self.max_pages}): {url}")

            html = await self.fetch_page(session, url)
            if html:
                title, text = self.clean_text(html)
                
                # Only save pages with useful content
                if len(text) > 100:
                    self.pages_scraped += 1
                    data = {
                        "url": url,
                        "title": title,
                        "text": text,
                        "timestamp": int(time.time())
                    }
                    await self.save_to_jsonl(data)

                    # Extract links and add them to the queue
                    links = self.extract_links(html, url)
                    for link in links:
                        if link not in self.visited:
                            self.queue.put_nowait(link)

            self.queue.task_done()
            # Yield control to allow other tasks to run
            await asyncio.sleep(0.01)

    async def run(self):
        """Starts the asynchronous crawler."""
        logger.info(f"Starting crawler. Max pages: {self.max_pages}, Concurrency: {self.max_concurrency}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(self.worker(session)) for _ in range(self.max_concurrency)]
            await asyncio.gather(*workers)
            
        logger.info(f"Crawling completed. Saved {self.pages_scraped} pages to {self.output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinnyCrawler - A polite, high-speed asynchronous web crawler")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of pages to scrape")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent worker tasks")
    parser.add_argument("--delay", type=float, default=1.0, help="Politeness delay in seconds between requests to the same domain")
    parser.add_argument("--output", type=str, default="data/scraped_data.jsonl", help="Path to write the JSONL file")
    args = parser.parse_args()

    # Pre-defined list of diverse starting points (Seeds)
    seeds = [
        "https://en.wikipedia.org/wiki/Special:Random",  # Random English Wikipedia page
        "https://de.wikipedia.org/wiki/Spezial:Zuf%C3%A4llige_Seite",  # Random German Wikipedia page
        "https://news.ycombinator.com/",                # Hacker News (lots of tech links)
        "https://www.gutenberg.org/",                   # Project Gutenberg (free books)
        "https://www.bbc.com/news"                      # BBC News
    ]

    crawler = FinnyCrawler(
        seed_urls=seeds,
        output_file=args.output,
        max_pages=args.limit,
        max_concurrency=args.concurrency,
        delay=args.delay
    )

    asyncio.run(crawler.run())
