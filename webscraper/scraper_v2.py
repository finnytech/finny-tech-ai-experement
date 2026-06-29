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
logger = logging.getLogger("FinnyCrawlerV2")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
]

class FinnyKnowledgeCrawler:
    def __init__(self, seed_urls, output_file, max_pages=100000, max_concurrency=40, delay_per_domain=0.5):
        self.seed_urls = seed_urls
        self.output_file = output_file
        self.max_pages = max_pages
        self.max_concurrency = max_concurrency
        self.delay_per_domain = delay_per_domain
        
        self.visited = set()
        self.pages_scraped = 0
        
        self.domain_queues = {}
        self.domain_visited = {}
        self.last_fetch_time = {}
        self.active_domains = set()
        
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

    def parse_wikipedia(self, html, url):
        """Extracts concepts and their definitions from Wikipedia articles."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Get the main page title
        title_el = soup.find("h1", id="firstHeading")
        if not title_el:
            return None
        concept = title_el.get_text().strip()
        
        # We only want actual articles, skip special pages
        if ":" in concept and not concept.startswith("Category:"):
            return None

        # Extract the introduction (paragraphs before the first TOC or H2 section)
        body = soup.find("div", id="mw-content-text")
        if not body:
            return None
            
        paragraphs = []
        for child in body.find_all("p", recursive=True):
            # Skip empty paragraphs or coordinates
            p_text = child.get_text().strip()
            if len(p_text) > 40:
                paragraphs.append(p_text)
            if len(paragraphs) >= 3:  # Keep first 3 paragraphs as the definition
                break
                
        if not paragraphs:
            return None
            
        definition = "\n".join(paragraphs)
        
        # Clean up citation marks like [1], [2], etc.
        definition = re.sub(r'\[\d+\]', '', definition)
        
        return {
            "type": "definition",
            "concept": concept,
            "text": definition,
            "url": url,
            "timestamp": int(time.time())
        }

    def parse_stackexchange(self, html, url):
        """Extracts questions and high-quality answers from StackExchange sites."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Get the question title
        title_el = soup.find("a", class_="question-hyperlink")
        if not title_el:
            title_el = soup.find("h1")
        if not title_el:
            return None
        question = title_el.get_text().strip()

        # Find the best answer (accepted answer first, or highest voted one)
        # Class "accepted-answer" marks the accepted one
        answer_div = soup.find("div", class_="accepted-answer")
        if not answer_div:
            # Fallback to the first answer div in the list
            answer_div = soup.find("div", class_="answer")
            
        if not answer_div:
            return None
            
        # Get the body of the answer
        answer_body = answer_div.find("div", class_="js-post-body")
        if not answer_body:
            return None
            
        # Remove code blocks if they are too long or nested, but keep simple explanations
        # Get text
        answer_text = answer_body.get_text(separator=" ").strip()
        
        # Clean up whitespace
        lines = (line.strip() for line in answer_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        answer_clean = "\n".join(chunk for chunk in chunks if chunk)

        if len(answer_clean) < 100:
            return None

        return {
            "type": "qa",
            "question": question,
            "text": answer_clean,
            "url": url,
            "timestamp": int(time.time())
        }

    def extract_links(self, html_content, base_url, target_domain):
        """Extracts internal links to continue crawling."""
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            if parsed.scheme in ("http", "https") and parsed.netloc == target_domain:
                # Exclude administrative pages on wikipedia
                if "wikipedia.org" in target_domain:
                    if any(x in parsed.path for x in ["/wiki/Special:", "/wiki/Help:", "/wiki/Wikipedia:", "/wiki/Talk:", "/wiki/File:"]):
                        continue
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.append(clean_url)
        return links

    async def fetch_page(self, session, url):
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        try:
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    return await response.text()
        except Exception:
            pass
        return None

    async def save_to_jsonl(self, data):
        async with self.write_lock:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def worker(self, session):
        while self.pages_scraped < self.max_pages:
            target_domain = None
            for domain, queue in self.domain_queues.items():
                if not queue.empty() and domain not in self.active_domains:
                    last_time = self.last_fetch_time.get(domain, 0)
                    if time.time() - last_time >= self.delay_per_domain:
                        target_domain = domain
                        break
            
            if not target_domain:
                await asyncio.sleep(0.1)
                continue
                
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
            
            self.last_fetch_time[target_domain] = time.time()
            html = await self.fetch_page(session, url)
            
            if html:
                parsed_data = None
                if "wikipedia.org" in target_domain:
                    parsed_data = self.parse_wikipedia(html, url)
                elif "stackexchange.com" in target_domain or "stackoverflow.com" in target_domain:
                    parsed_data = self.parse_stackexchange(html, url)
                
                if parsed_data:
                    self.pages_scraped += 1
                    await self.save_to_jsonl(parsed_data)
                    
                    if self.pages_scraped % 100 == 0:
                        logger.info(f"Progress: {self.pages_scraped} knowledge units scraped total.")
                        
                # Extract and enqueue new links
                links = self.extract_links(html, url, target_domain)
                for link in links:
                    if link not in visited:
                        queue.put_nowait(link)
                            
            queue.task_done()
            self.active_domains.remove(target_domain)
            await asyncio.sleep(0.01)

    async def run(self):
        logger.info(f"Starting FinnyCrawlerV2. Target: {self.max_pages} Q&A and Concept units.")
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(self.worker(session)) for _ in range(self.max_concurrency)]
            await asyncio.gather(*workers)
            
        logger.info(f"Finished. Scraped {self.pages_scraped} units to {self.output_file}.")

import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinnyCrawler V2 - Concept Definition and Q&A Extractor")
    parser.add_argument("--limit", type=int, default=50000, help="Maximum units to scrape")
    parser.add_argument("--concurrency", type=int, default=40, help="Total parallel workers")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay per domain")
    parser.add_argument("--output", type=str, default="data/knowledge_data.jsonl", help="Output path")
    args = parser.parse_args()

    # Seeds specifically chosen for high-quality definitions and explanations (Wikis & StackExchange)
    seeds = [
        # Concept/Definition Seeds
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Computer_science",
        "https://en.wikipedia.org/wiki/Physics",
        "https://en.wikipedia.org/wiki/Mathematics",
        "https://en.wikipedia.org/wiki/Philosophy",
        "https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz",
        "https://de.wikipedia.org/wiki/Informatik",
        "https://de.wikipedia.org/wiki/Physik",
        
        # Q&A / Explanation Seeds
        "https://stackoverflow.com/questions?tab=Active",
        "https://cs.stackexchange.com/questions",
        "https://physics.stackexchange.com/questions",
        "https://math.stackexchange.com/questions",
        "https://philosophy.stackexchange.com/questions",
        "https://ai.stackexchange.com/questions",
        "https://codereview.stackexchange.com/questions",
        "https://datascience.stackexchange.com/questions"
    ]

    crawler = FinnyKnowledgeCrawler(
        seed_urls=seeds,
        output_file=args.output,
        max_pages=args.limit,
        max_concurrency=args.concurrency,
        delay_per_domain=args.delay
    )

    asyncio.run(crawler.run())
