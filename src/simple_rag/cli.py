"""Rich interactive Command Line Interface for Simple RAG."""

import argparse
from pathlib import Path
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from simple_rag.pipeline import RAGPipeline
from simple_rag.embeddings import get_embedding_model
from simple_rag.llm import get_llm
from simple_rag.vector_store import SimpleVectorStore

console = Console()
DEFAULT_INDEX_PATH = "rag_index.json"


def get_default_pipeline(index_path: str = DEFAULT_INDEX_PATH, provider: str = "auto") -> RAGPipeline:
    """Initialize pipeline and load persisted index if present."""
    embedder = get_embedding_model(provider=provider)
    llm = get_llm(provider=provider)
    store = SimpleVectorStore()

    p = Path(index_path)
    if p.exists():
        store.load(p)

    return RAGPipeline(embedding_model=embedder, llm=llm, vector_store=store)


def cmd_ingest(args):
    """Ingest documents into the vector store."""
    console.print(Panel.fit("[bold cyan]Simple RAG Document Ingestion[/bold cyan]"))
    pipeline = get_default_pipeline(args.index, provider=args.provider)

    target_path = Path(args.path)
    if not target_path.exists():
        console.print(f"[bold red]Error: Path '{target_path}' does not exist.[/bold red]")
        sys.exit(1)

    with console.status(f"[bold green]Ingesting from {target_path}..."):
        if target_path.is_dir():
            stats = pipeline.ingest_directory(target_path)
        else:
            stats = pipeline.ingest_file(target_path)

        pipeline.save_index(args.index)

    table = Table(title="Ingestion Summary", border_style="cyan")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="green")
    table.add_row("Source Path", str(target_path))
    table.add_row("Documents Ingested", str(stats["documents_loaded"]))
    table.add_row("Chunks Created", str(stats["chunks_created"]))
    table.add_row("Total Chunks in Index", str(stats["total_chunks_in_store"]))
    table.add_row("Saved Index File", str(Path(args.index).resolve()))
    console.print(table)


def cmd_query(args):
    """Execute a single query."""
    pipeline = get_default_pipeline(args.index, provider=args.provider)
    if pipeline.vector_store.count == 0:
        console.print("[bold yellow]Warning:[/bold yellow] Vector store is empty! Run `ingest` first.")
        sys.exit(1)

    console.print(f"\n[bold magenta]Query:[/bold magenta] {args.query}\n")

    with console.status("[bold cyan]Retrieving context & generating answer..."):
        resp = pipeline.query(args.query, top_k=args.top_k)

    # Display Answer
    console.print(Panel(Markdown(resp.answer), title=f"[bold green]Answer ({resp.model_used})[/bold green]", border_style="green"))

    # Display Sources Table
    table = Table(title=f"Retrieved Sources (Top {len(resp.sources)})", border_style="blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Source", style="cyan")
    table.add_column("Similarity", style="bold yellow", justify="right")
    table.add_column("Snippet Preview", style="white")

    for i, res in enumerate(resp.sources, 1):
        filename = res.chunk.metadata.get("filename", "Unknown")
        preview = res.chunk.content.replace("\n", " ")[:120] + "..."
        score_pct = f"{res.score * 100:.1f}%"
        table.add_row(str(i), filename, score_pct, preview)

    console.print(table)
    console.print(
        f"[dim]Retrieval: {resp.retrieval_time_ms:.1f}ms | Generation: {resp.generation_time_ms:.1f}ms[/dim]\n"
    )


def cmd_chat(args):
    """Interactive streaming chat session."""
    pipeline = get_default_pipeline(args.index, provider=args.provider)
    console.print(
        Panel.fit(
            "[bold green]Simple RAG Interactive Chat[/bold green]\n"
            f"[dim]Model: {pipeline.llm.model_name} | Indexed Chunks: {pipeline.vector_store.count}[/dim]\n"
            "[yellow]Type your questions, '/sources' to inspect top chunks, or 'exit' / 'quit' to leave.[/yellow]"
        )
    )

    last_sources = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        if user_input.lower() == "/sources":
            if not last_sources:
                console.print("[yellow]No sources from previous query.[/yellow]")
                continue
            for i, res in enumerate(last_sources, 1):
                fn = res.chunk.metadata.get("filename", "Unknown")
                console.print(
                    Panel(
                        f"[bold cyan]Source:[/bold cyan] {fn} (Score: {res.score*100:.1f}%)\n\n{res.chunk.content}",
                        title=f"Source Chunk #{i}",
                        border_style="cyan",
                    )
                )
            continue

        if user_input.lower() == "/help":
            console.print("Commands: 'exit'/'quit' to leave, '/sources' to see last retrieved chunks.")
            continue

        console.print("[bold green]Assistant:[/bold green] ", end="")
        try:
            stream_iter, sources = pipeline.stream_query(user_input, top_k=args.top_k)
            last_sources = sources
            full_text = ""
            for token in stream_iter:
                sys.stdout.write(token)
                sys.stdout.flush()
                full_text += token
            console.print()

            if sources:
                src_names = list({s.chunk.metadata.get("filename", "doc") for s in sources})
                console.print(f"[dim](Referenced: {', '.join(src_names)})[/dim]")
        except Exception as e:
            console.print(f"\n[bold red]Error during query:[/bold red] {e}")


def cmd_serve(args):
    """Launch FastAPI Web UI."""
    try:
        import uvicorn
        from simple_rag.web import app, init_web_pipeline
    except ImportError as e:
        console.print(f"[bold red]Error loading web dependencies: {e}[/bold red]")
        sys.exit(1)

    init_web_pipeline(index_path=args.index, provider=args.provider)
    console.print(f"[bold green]Starting Web UI at http://{args.host}:{args.port}[/bold green]")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Path to vector store index JSON")
    common_parser.add_argument("--provider", default="auto", choices=["auto", "openai", "local"], help="LLM and embedding provider")

    parser = argparse.ArgumentParser(
        prog="simple-rag",
        description="Simple RAG: An educational, extensible Retrieval-Augmented Generation CLI",
        parents=[common_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    parser_ingest = subparsers.add_parser("ingest", help="Ingest a directory or file into vector store", parents=[common_parser])
    parser_ingest.add_argument("path", help="Directory or file path to ingest")

    # Query command
    parser_query = subparsers.add_parser("query", help="Ask a one-off question", parents=[common_parser])
    parser_query.add_argument("query", help="The question to ask")
    parser_query.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")

    # Chat command
    parser_chat = subparsers.add_parser("chat", help="Start an interactive chat session", parents=[common_parser])
    parser_chat.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")

    # Serve command
    parser_serve = subparsers.add_parser("serve", help="Launch FastAPI Web UI", parents=[common_parser])
    parser_serve.add_argument("--host", default="127.0.0.1", help="Host address")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
