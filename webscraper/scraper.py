import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import json
import os
import argparse
import random
import time
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("FinnyCrawler")

# Rotate user agents to look like real browser traffic
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edge/120.0.0.0"
]

class FinnyCrawler:
    def __init__(self, seed_urls, output_file, max_pages=100000, max_concurrency=40, delay_per_domain=0.5):
        self.seed_urls = seed_urls
        self.output_file = output_file
        self.max_pages = max_pages
        self.max_concurrency = max_concurrency
        self.delay_per_domain = delay_per_domain
        
        # Global visited set
        self.visited = set()
        self.pages_scraped = 0
        
        # Domain management to prevent bans:
        # We store queues separately for each domain.
        self.domain_queues = {}
        self.domain_visited = {}
        
        # Track when we last fetched from each domain
        self.last_fetch_time = {}
        # Track domains currently being requested to ensure concurrency-per-domain is strictly 1
        self.active_domains = set()
        
        # Write lock for JSONL
        self.write_lock = asyncio.Lock()

        # Initialize seeds
        for url in self.seed_urls:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                if domain not in self.domain_queues:
                    self.domain_queues[domain] = asyncio.Queue()
                    self.domain_visited[domain] = set()
                self.domain_queues[domain].put_nowait(url)

    def clean_text(self, html_content):
        """Extracts and filters clean text from HTML."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove junk elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            element.extract()
            
        text = soup.get_text(separator=" ")
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        title = soup.title.string.strip() if soup.title else ""
        return title, clean_text

    def extract_links(self, html_content, base_url, target_domain):
        """Extracts internal links to stay on the same website per worker loop."""
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            # Stay on the same domain
            if parsed.scheme in ("http", "https") and parsed.netloc == target_domain:
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.append(clean_url)
        return links

    async def fetch_page(self, session, url):
        """Fetches the page HTML safely with user-agent rotation."""
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        try:
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:
                    logger.warning(f"Rate limit (429) on: {url}. Backing off.")
        except Exception:
            pass
        return None

    async def save_to_jsonl(self, data):
        """Thread-safe JSONL writer."""
        async with self.write_lock:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def worker(self, session):
        """Worker that cycles through domains to fetch pages, ensuring no domain overload."""
        while self.pages_scraped < self.max_pages:
            # Find a domain queue that has items and is not currently active
            target_domain = None
            for domain, queue in self.domain_queues.items():
                if not queue.empty() and domain not in self.active_domains:
                    # Check delay/cooldown
                    last_time = self.last_fetch_time.get(domain, 0)
                    if time.time() - last_time >= self.delay_per_domain:
                        target_domain = domain
                        break
            
            if not target_domain:
                # No domain is ready or has work. Wait briefly and try again.
                await asyncio.sleep(0.1)
                continue
                
            # Lock the domain to this worker
            self.active_domains.add(target_domain)
            queue = self.domain_queues[target_domain]
            visited = self.domain_visited[target_domain]
            
            try:
                url = queue.get_nowait()
            except asyncio.QueueEmpty:
                self.active_domains.remove(target_domain)
                continue

            if url in visited:
                queue.task_done()
                self.active_domains.remove(target_domain)
                continue
                
            visited.add(url)
            
            # Fetch
            self.last_fetch_time[target_domain] = time.time()
            html = await self.fetch_page(session, url)
            
            if html:
                title, text = self.clean_text(html)
                if len(text) > 200:
                    self.pages_scraped += 1
                    data = {
                        "url": url,
                        "title": title,
                        "text": text,
                        "timestamp": int(time.time())
                    }
                    await self.save_to_jsonl(data)
                    
                    if self.pages_scraped % 100 == 0:
                        logger.info(f"Progress: {self.pages_scraped} pages scraped total.")
                        
                    # Extract and enqueue new links
                    links = self.extract_links(html, url, target_domain)
                    for link in links:
                        if link not in visited:
                            queue.put_nowait(link)
                            
            queue.task_done()
            # Release domain
            self.active_domains.remove(target_domain)
            # Yield control
            await asyncio.sleep(0.01)

    async def run(self):
        logger.info(f"Starting crawler. Max Concurrency: {self.max_concurrency}. Delay per domain: {self.delay_per_domain}s")
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(self.worker(session)) for _ in range(self.max_concurrency)]
            await asyncio.gather(*workers)
            
        logger.info(f"Finished. Scraped {self.pages_scraped} pages to {self.output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinnyCrawler Pro - Safe & Ultra-Fast Multi-Domain Scraper")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum pages to scrape")
    parser.add_argument("--concurrency", type=int, default=40, help="Total parallel workers")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests to the same domain")
    parser.add_argument("--output", type=str, default="data/training_data.jsonl", help="Output path")
    args = parser.parse_args()

    # Highly diverse list of websites to distribute load and speed up crawls safely
    seeds = [
        # Wikis
        "https://en.wikipedia.org/wiki/Main_Page",
        "https://de.wikipedia.org/wiki/Wikipedia:Hauptseite",
        "https://fr.wikipedia.org/wiki/Portail:Accueil",
        "https://es.wikipedia.org/wiki/Wikipedia:Portada",
        # Tech & News
        "https://news.ycombinator.com/",
        "https://www.bbc.com/news",
        "https://www.spiegel.de/",
        "https://www.heise.de/",
        "https://www.nytimes.com/",
        "https://www.cnn.com/",
        "https://www.reuters.com/",
        "https://www.theguardian.com/international",
        "https://www.technologyreview.com/",
        "https://www.wired.com/",
        "https://www.bloomberg.com/",
        # Science & Education
        "https://www.nature.com/",
        "https://www.scientificamerican.com/",
        "https://www.nasa.gov/",
        "https://www.britannica.com/",
        "https://www.nationalgeographic.com/",
        # Books & General
        "https://www.gutenberg.org/",
        "https://www.gutenberg.org/ebooks/search/?sort_order=downloads",
        "https://archive.org/",
        "https://www.gutenberg.org/ebooks/bookshelves"
    ]

    crawler = FinnyCrawler(
        seed_urls=seeds,
        output_file=args.output,
        max_pages=args.limit,
        max_concurrency=args.concurrency,
        delay_per_domain=args.delay
    )

    asyncio.run(crawler.run())
