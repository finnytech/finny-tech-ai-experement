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

# Rotate user agents to avoid detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edge/120.0.0.0"
]

class FinnyCrawler:
    def __init__(self, seed_urls, output_file, max_pages=100000, concurrency_per_domain=5, total_concurrency=40):
        self.seed_urls = seed_urls
        self.output_file = output_file
        self.max_pages = max_pages
        self.total_concurrency = total_concurrency
        self.concurrency_per_domain = concurrency_per_domain
        
        # Track state
        self.visited = set()
        self.pages_scraped = 0
        
        # We group URLs and queues by domain to crawl multiple websites in parallel
        self.domain_queues = {}
        self.domain_visited = {}
        self.active_workers_per_domain = {}
        
        # Write lock for JSONL output file
        self.write_lock = asyncio.Lock()

        # Initialize queues for each seed domain
        for url in self.seed_urls:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain not in self.domain_queues:
                self.domain_queues[domain] = asyncio.Queue()
                self.domain_visited[domain] = set()
                self.active_workers_per_domain[domain] = 0
            
            self.domain_queues[domain].put_nowait(url)

    def clean_text(self, html_content):
        """Extracts and filters clean text from HTML, removing headers, footers, scripts, styles."""
        soup = BeautifulSoup(html_content, "lxml" if "lxml" in html_content else "html.parser")
        
        # Remove non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            element.extract()
            
        # Get text
        text = soup.get_text(separator=" ")
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        title = soup.title.string.strip() if soup.title else ""
        return title, clean_text

    def extract_links(self, html_content, base_url, target_domain):
        """Extracts valid internal links to stay on the same website for deep scraping."""
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            # Stay on the same website/domain to keep workers separated
            if parsed.scheme in ("http", "https") and parsed.netloc == target_domain:
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.append(clean_url)
        return links

    async def fetch_page(self, session, url):
        """Fetches the page HTML instantly without artificial delay."""
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        try:
            # Short timeout of 5 seconds to skip slow pages
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    return await response.text()
        except Exception:
            pass
        return None

    async def save_to_jsonl(self, data):
        """Thread-safe writing of scraped page data."""
        async with self.write_lock:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def domain_worker(self, session, domain):
        """Worker dedicated to a single domain, crawling as fast as possible."""
        queue = self.domain_queues[domain]
        visited = self.domain_visited[domain]

        while self.pages_scraped < self.max_pages:
            try:
                # Retrieve URL from this domain's queue
                url = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # If queue is empty for 2 seconds, wait a bit or exit if no new links are coming
                await asyncio.sleep(1.0)
                if queue.empty():
                    break
                continue

            if url in visited:
                queue.task_done()
                continue

            visited.add(url)
            
            # Fetch the page immediately (no sleep delay!)
            html = await self.fetch_page(session, url)
            if html:
                title, text = self.clean_text(html)
                
                # Only keep articles with substantial text content
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

                    # Extract new links for this domain
                    links = self.extract_links(html, url, domain)
                    for link in links:
                        if link not in visited:
                            queue.put_nowait(link)

            queue.task_done()

    async def run(self):
        """Launches parallel domain workers."""
        logger.info(f"Starting crawler. Target: {self.max_pages} pages.")
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            # For each domain, spawn several parallel workers to crawl it concurrently
            for domain in self.domain_queues.keys():
                for _ in range(self.concurrency_per_domain):
                    tasks.append(asyncio.create_task(self.domain_worker(session, domain)))
            
            await asyncio.gather(*tasks)
            
        logger.info(f"Finished. Scraped {self.pages_scraped} pages to {self.output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinnyCrawler Ultra - Aggressive Parallel Web Scraper")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum pages to scrape")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency per domain")
    parser.add_argument("--output", type=str, default="data/training_data.jsonl", help="Output path")
    args = parser.parse_args()

    # Seed list with a wide range of content-heavy target sites
    seeds = [
        "https://en.wikipedia.org/wiki/Main_Page",
        "https://de.wikipedia.org/wiki/Wikipedia:Hauptseite",
        "https://news.ycombinator.com/",
        "https://www.gutenberg.org/",
        "https://www.bbc.com/news",
        "https://www.spiegel.de/",
        "https://www.heise.de/",
        "https://www.nytimes.com/",
        "https://www.cnn.com/",
        "https://www.reuters.com/",
        "https://www.theguardian.com/international"
    ]

    crawler = FinnyCrawler(
        seed_urls=seeds,
        output_file=args.output,
        max_pages=args.limit,
        concurrency_per_domain=args.concurrency
    )

    asyncio.run(crawler.run())
