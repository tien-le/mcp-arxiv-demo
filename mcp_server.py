import json
import os

import arxiv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP

PAPER_DIR = "papers"
JSON_FILENAME = "papers_info_tien.json"

# Get port from environment variable (Render sets this) or default to 8001
PORT = int(os.environ.get("PORT", 8001))

# Init FastMCP Server
mcp = FastMCP(name="research", port=PORT)

# Wrap sse_app to add CORS middleware
original_sse_app = mcp.sse_app


def sse_app_with_cors(mount_path: str | None = None):
    """Wrapper for sse_app that adds CORS middleware."""
    app = original_sse_app(mount_path)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods including OPTIONS
        allow_headers=["*"],  # Allow all headers
    )
    return app


mcp.sse_app = sse_app_with_cors


##########################
# Search relevant papers
##########################


@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> list[str]:
    """Search papers on arXiv based on given topic."""
    client = arxiv.Client()
    search_query = arxiv.Search(
        query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    relevant_papers = client.results(search_query)

    # Directory
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, JSON_FILENAME)

    # Load existing papers info
    papers_info = {}
    try:
        with open(file_path, "r", encoding="utf8") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Raise exception, due to: ", e)

    # Add papers info
    paper_ids = []
    for paper in relevant_papers:
        paper_ids.append(paper.get_short_id())  # type: ignore
        paper_info = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
        }
        papers_info[paper.get_short_id()] = paper_info

    # Save to file
    with open(file_path, mode="w", encoding="utf8") as json_file:
        json.dump(papers_info, json_file, indent=2)
    print(f"Saved to file: {file_path}")
    return paper_ids


# Unittest
# search_papers("LLM")

#######################
# Extract information
#######################


@mcp.tool()
def extract_info(paper_id: str) -> str:
    """Search information of paper whose id is given."""
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, JSON_FILENAME)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, mode="r", encoding="utf8") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error when reading file {file_path}: {e}")
                    continue
    return f"Not Found any information related to paper id={paper_id}"


##########################
# Resource management
##########################
@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """List all available topic folders in the papers directory."""
    folders: list[str] = []
    if not os.path.exists(PAPER_DIR):
        return f"Folder {PAPER_DIR} does not exist"
    for topic_dir in os.listdir(PAPER_DIR):
        topic_path = os.path.join(PAPER_DIR, topic_dir)
        if os.path.isdir(topic_path):
            papers_file = os.path.join(topic_path, JSON_FILENAME)
            if os.path.isfile(papers_file):
                folders.append(topic_dir)
    content = "# Available Topics\n\n"
    if folders:
        for topic_dir in folders:
            content += f"- {topic_dir}\n"
            content += f"\nUse @{topic_dir} to access papers in that topic.\n"
    else:
        content += "No topics found.\n"
    return content


@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """Get detailed information about papers on a specific topic."""
    topic_dir = topic.lower().replace(" ", "_")
    papers_file = os.path.join(PAPER_DIR, topic_dir, JSON_FILENAME)
    if not os.path.isfile(papers_file):
        return f"# No papers found for topic: {topic}\n\nTry searching for papers on this topic first."
    try:
        with open(papers_file, "r", encoding="utf8") as json_file:
            papers_info = json.load(json_file)
        content = f"# Papers on {topic.replace('_', ' ').title()}\n\n"
        content += f"Total papers: {len(papers_info)}\n\n"
        for paper_id, paper_info in papers_info.items():
            content += f"## {paper_info['title']}\n"
            content += f"- **Paper ID**: {paper_id}\n"
            content += f"- **Authors**: {', '.join(paper_info['authors'])}\n"
            content += f"- **Published**: {paper_info['published']}\n"
            content += (
                f"- **PDF URL**: [{paper_info['pdf_url']}]({paper_info['pdf_url']})\n\n"
            )
            content += f"### Summary\n{paper_info['summary'][:500]}...\n\n"
            content += "---\n\n"
        return content
    except json.JSONDecodeError:
        return f"# Error reading papers data for {topic}\n\nThe papers data file is corrupted."


@mcp.prompt()
def generate_search_prompt(topic: str, max_results: int = 5) -> str:
    """Generate a prompt for Claude to find and discuss academic papers on a specific topic."""
    return f"""Search for {max_results} academic papers about '{topic}' using the search_papers tool. Follow these instructions:
    1. First, search for papers using search_papers(topic='{topic}', max_results={max_results})
    2. For each paper found, extract and organize the following information:
       - Paper title
       - Authors
       - Publication date
       - Brief summary of the key findings
       - Main contributions or innovations
       - Methodologies used
       - Relevance to the topic '{topic}'

    3. Provide a comprehensive summary that includes:
       - Overview of the current state of research in '{topic}'
       - Common themes and trends across the papers
       - Key research gaps or areas for future investigation
       - Most impactful or influential papers in this area

    4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.

    Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""


if __name__ == "__main__":
    print("Running MCP server...")
    print(f"MCP Server will be available at: http://0.0.0.0:{PORT}/sse")
    print("Make sure this server is running before starting the MCP Inspector!")
    # Get the FastAPI app with CORS middleware
    app = mcp.sse_app()
    # Run with uvicorn, binding to 0.0.0.0 to allow external connections (required for Render)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
