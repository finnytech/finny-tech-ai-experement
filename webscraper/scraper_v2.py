import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import json
import os
import argparse
import random
import time
import re
import logging
import concurrent.futures

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

# Words that indicate a question/explanation structure
QUESTION_WORDS_EN = r"\b(how|why|what|who|where|which|can|should|is|are|does|do|did|explain|definition)\b"
QUESTION_WORDS_DE = r"\b(wie|warum|was|wer|wo|welche|kann|soll|ist|sind|erkläre|definition|bedeutung)\b"
QUESTION_PATTERN = re.compile(
    rf"({QUESTION_WORDS_EN}|{QUESTION_WORDS_DE})", 
    re.IGNORECASE
)

def process_html_in_worker(html_content, url, target_domain):
    """
    Runs in a separate process worker to offload CPU-bound HTML parsing 
    and BeautifulSoup extraction from the async event loop.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    extracted_units = []
    
    # 1. Parse content based on domain type
    if "wikipedia.org" in target_domain:
        # Wikipedia parser
        title_el = soup.find("h1", id="firstHeading")
        if title_el:
            concept = title_el.get_text().strip()
            if not (":" in concept and not concept.startswith("Category:")):
                body = soup.find("div", id="mw-content-text")
                if body:
                    paragraphs = []
                    for child in body.find_all("p", recursive=True):
                        p_text = child.get_text().strip()
                        if len(p_text) > 50:
                            paragraphs.append(p_text)
                        if len(paragraphs) >= 3:
                            break
                    if paragraphs:
                        definition = "\n".join(paragraphs)
                        definition = re.sub(r'\[\d+\]', '', definition)  # Remove citations
                        extracted_units.append({
                            "type": "definition",
                            "concept": concept,
                            "text": definition,
                            "url": url,
                            "timestamp": int(time.time())
                        })
                        
    elif "stackexchange.com" in target_domain or "stackoverflow.com" in target_domain:
        # StackExchange parser
        title_el = soup.find("a", class_="question-hyperlink") or soup.find("h1")
        if title_el:
            question = title_el.get_text().strip()
            answer_div = soup.find("div", class_="accepted-answer") or soup.find("div", class_="answer")
            if answer_div:
                answer_body = answer_div.find("div", class_="js-post-body")
                if answer_body:
                    answer_text = answer_body.get_text(separator=" ").strip()
                    lines = (line.strip() for line in answer_text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    answer_clean = "\n".join(chunk for chunk in chunks if chunk)
                    if len(answer_clean) > 100:
                        extracted_units.append({
                            "type": "qa",
                            "question": question,
                            "text": answer_clean,
                            "url": url,
                            "timestamp": int(time.time())
                        })
    else:
        # General Web Page Parser (Q&A and Definitions Heuristics)
        title_el = soup.find("title")
        page_title = title_el.get_text().strip() if title_el else ""
        
        # Look for headings followed by paragraphs (Q&A structure)
        headings = soup.find_all(["h1", "h2", "h3"])
        for h in headings:
            heading_text = h.get_text().strip()
            is_question = heading_text.endswith("?") or QUESTION_PATTERN.search(heading_text)
            if is_question and 10 < len(heading_text) < 150:
                paragraphs = []
                sibling = h.find_next_sibling()
                while sibling and sibling.name not in ["h1", "h2", "h3"]:
                    if sibling.name == "p":
                        p_text = sibling.get_text().strip()
                        if len(p_text) > 40:
                            paragraphs.append(p_text)
                    sibling = sibling.find_next_sibling()
                    if len(paragraphs) >= 3:
                        break
                if paragraphs:
                    extracted_units.append({
                        "type": "qa",
                        "question": heading_text,
                        "text": "\n".join(paragraphs),
                        "url": url,
                        "timestamp": int(time.time())
                    })
                    
        # Fallback to general concept definition using page title
        if not extracted_units and len(page_title) > 5:
            clean_title = re.split(r'\s+[-|•]\s+', page_title)[0].strip()
            paragraphs = []
            for p in soup.find_all("p"):
                p_text = p.get_text().strip()
                if len(p_text) > 60:
                    paragraphs.append(p_text)
                if len(paragraphs) >= 2:
                    break
            if paragraphs:
                extracted_units.append({
                    "type": "definition",
                    "concept": clean_title,
                    "text": "\n".join(paragraphs),
                    "url": url,
                    "timestamp": int(time.time())
                })

    # 2. Extract links
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor['href']
        full_url = urljoin(url, href)
        parsed = urlparse(full_url)
        if parsed.scheme in ("http", "https") and parsed.netloc == target_domain:
            # Exclude administrative pages on wikipedia
            if "wikipedia.org" in target_domain:
                if any(x in parsed.path for x in ["/wiki/Special:", "/wiki/Help:", "/wiki/Wikipedia:", "/wiki/Talk:", "/wiki/File:"]):
                    continue
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            links.append(clean_url)

    return links, extracted_units


class FinnyKnowledgeCrawler:
    def __init__(self, seed_urls, output_file, max_pages=100000, max_concurrency=40, delay_per_domain=0.5, num_cores=8):
        self.seed_urls = seed_urls
        self.output_file = output_file
        self.max_pages = max_pages
        self.max_concurrency = max_concurrency
        self.delay_per_domain = delay_per_domain
        self.num_cores = num_cores
        
        self.visited = set()
        self.pages_scraped = 0
        
        self.domain_queues = {}
        self.domain_visited = {}
        self.last_fetch_time = {}
        self.active_domains = set()
        
        self.write_lock = asyncio.Lock()
        
        # Initialize the ProcessPoolExecutor to run CPU-bound parsing on all available CPU cores
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores)

        # Initialize seeds
        for url in self.seed_urls:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                if domain not in self.domain_queues:
                    self.domain_queues[domain] = asyncio.Queue()
                    self.domain_visited[domain] = set()
                self.domain_queues[domain].put_nowait(url)

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
        loop = asyncio.get_running_loop()
        
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
                try:
                    # Offload the BeautifulSoup parsing and link extraction to the process pool!
                    # This runs concurrently on your 8 CPU cores.
                    links, extracted_data_list = await loop.run_in_executor(
                        self.executor, process_html_in_worker, html, url, target_domain
                    )
                    
                    # Save extracted units
                    for data in extracted_data_list:
                        self.pages_scraped += 1
                        await self.save_to_jsonl(data)
                        
                        if self.pages_scraped % 100 == 0:
                            logger.info(f"Progress: {self.pages_scraped} knowledge units scraped.")

                    # Enqueue new links
                    for link in links:
                        parsed_link = urlparse(link)
                        domain = parsed_link.netloc
                        if domain:
                            if domain not in self.domain_queues:
                                self.domain_queues[domain] = asyncio.Queue()
                                self.domain_visited[domain] = set()
                            
                            if link not in self.domain_visited[domain]:
                                self.domain_queues[domain].put_nowait(link)
                except Exception as e:
                    logger.warning(f"Error parsing page {url}: {e}")
                            
            queue.task_done()
            self.active_domains.remove(target_domain)
            await asyncio.sleep(0.01)

    async def run(self):
        logger.info(f"Starting Multi-Core FinnyCrawlerV2. Cores: {self.num_cores}. Target: {self.max_pages} units.")
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(self.worker(session)) for _ in range(self.max_concurrency)]
            await asyncio.gather(*workers)
            
        logger.info(f"Finished. Scraped {self.pages_scraped} knowledge units to {self.output_file}.")
        self.executor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinnyCrawler V2 - Multi-Core General Knowledge Extractor")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum units to scrape")
    parser.add_argument("--concurrency", type=int, default=40, help="Total parallel workers")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay per domain")
    parser.add_argument("--cores", type=int, default=8, help="Number of CPU cores to utilize")
    parser.add_argument("--output", type=str, default="data/knowledge_data.jsonl", help="Output path")
    args = parser.parse_args()

    # Highly diverse seeds to start crawling the general web randomly
    seeds = [
        "https://en.wikipedia.org/wiki/Special:Random",
        "https://de.wikipedia.org/wiki/Spezial:Zuf%C3%A4llige_Seite",
        "https://news.ycombinator.com/",
        "https://www.bbc.com/news",
        "https://www.spiegel.de/",
        "https://www.gutenberg.org/",
        "https://www.nytimes.com/",
        "https://www.nature.com/",
        "https://www.britannica.com/",
        "https://archive.org/"
    ]

    crawler = FinnyKnowledgeCrawler(
        seed_urls=seeds,
        output_file=args.output,
        max_pages=args.limit,
        max_concurrency=args.concurrency,
        delay_per_domain=args.delay,
        num_cores=args.cores
    )

    asyncio.run(crawler.run())
