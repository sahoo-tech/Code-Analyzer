

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from analyzer.engine import CodeAnalyzer
from analyzer.config import ConfigManager
from analyzer.logging_config import configure_logging
from analyzer import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="code-analyzer",
        description="Enterprise-level Python Code Analyzer for AI-enhanced code understanding",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"code-analyzer {__version__}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze code")
    analyze_parser.add_argument(
        "path",
        help="File or directory to analyze",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)",
    )
    analyze_parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "summary"],
        default="json",
        help="Output format (default: json)",
    )
    analyze_parser.add_argument(
        "--no-security",
        action="store_true",
        help="Skip security analysis",
    )
    analyze_parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Skip pattern detection",
    )
    analyze_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query codebase")
    query_parser.add_argument(
        "path",
        help="File or directory to query",
    )
    query_parser.add_argument(
        "query",
        help="Natural language query",
    )
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Get code summary for AI")
    summary_parser.add_argument(
        "path",
        help="File or directory to summarize",
    )
    summary_parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Maximum tokens for summary (default: 8000)",
    )
    
    # Init config command
    init_parser = subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "-f", "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Config file format (default: yaml)",
    )
    
    # RAG subparser
    rag_parser = subparsers.add_parser(
        "rag", 
        help="RAG (Retrieval-Augmented Generation) commands"
    )
    rag_subparsers = rag_parser.add_subparsers(dest="rag_command", help="RAG commands")
    
    # RAG index command
    rag_index = rag_subparsers.add_parser("index", help="Index codebase for RAG")
    rag_index.add_argument(
        "path",
        help="Directory to index",
    )
    rag_index.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before indexing",
    )
    rag_index.add_argument(
        "--persist-dir",
        default=".analyzer_rag",
        help="Directory for persistent storage (default: .analyzer_rag)",
    )
    
    # RAG ask command
    rag_ask = rag_subparsers.add_parser("ask", help="Ask a question about the code")
    rag_ask.add_argument(
        "question",
        help="Question to ask about the codebase",
    )
    rag_ask.add_argument(
        "--path",
        help="Optional: Index this path first before asking",
    )
    rag_ask.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of code chunks to retrieve (default: 10)",
    )
    rag_ask.add_argument(
        "--persist-dir",
        default=".analyzer_rag",
        help="Directory for persistent storage",
    )
    
    # RAG search command
    rag_search = rag_subparsers.add_parser("search", help="Semantic search over code")
    rag_search.add_argument(
        "query",
        help="Search query",
    )
    rag_search.add_argument(
        "--path",
        help="Optional: Index this path first",
    )
    rag_search.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results (default: 10)",
    )
    rag_search.add_argument(
        "--type",
        choices=["class", "function", "method", "module"],
        help="Filter by entity type",
    )
    rag_search.add_argument(
        "--persist-dir",
        default=".analyzer_rag",
        help="Directory for persistent storage",
    )
    
    # RAG clear command
    rag_clear = rag_subparsers.add_parser("clear", help="Clear the RAG index")
    rag_clear.add_argument(
        "--persist-dir",
        default=".analyzer_rag",
        help="Directory for persistent storage",
    )
    
    # RAG stats command
    rag_stats = rag_subparsers.add_parser("stats", help="Show RAG index statistics")
    rag_stats.add_argument(
        "--persist-dir",
        default=".analyzer_rag",
        help="Directory for persistent storage",
    )
    
    return parser


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute analyze command."""
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1
    
    # Create analyzer
    config = ConfigManager.auto_discover(path.parent if path.is_file() else path)
    
    # Apply command-line overrides
    if args.no_security:
        config.config.security.check_sql_injection = False
        config.config.security.check_hardcoded_secrets = False
        config.config.security.check_dangerous_functions = False
    
    if args.no_patterns:
        config.config.patterns.detect_design_patterns = False
        config.config.patterns.detect_anti_patterns = False
        config.config.patterns.detect_code_smells = False
    
    if args.verbose:
        config.config.logging.level = "DEBUG"
    
    analyzer = CodeAnalyzer(config.config)
    
    # Analyze
    if path.is_file():
        result = analyzer.analyze_file(path)
    else:
        result = analyzer.analyze_directory(path)
    
    # Format output
    if args.format == "json":
        output = result.to_json()
    elif args.format == "markdown":
        output = analyzer.get_ai_summary(result, "markdown")
    else:
        summary = result.get_summary()
        output = format_summary(summary)
    
    # Write output
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Output written to: {args.output}")
    else:
        print(output)
    
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Execute query command."""
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1
    
    analyzer = CodeAnalyzer()
    
    if path.is_file():
        result = analyzer.analyze_file(path)
    else:
        result = analyzer.analyze_directory(path)
    
    query_result = analyzer.query(result, args.query)
    print(json.dumps(query_result, indent=2))
    
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Execute summary command."""
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1
    
    from analyzer.ai.summarizer import summarize_project, SummaryConfig
    from analyzer.parsers import FileParser
    
    parser = FileParser()
    
    if path.is_file():
        modules = [parser.parse_file(path)]
    else:
        modules = parser.parse_directory(path)
    
    config = SummaryConfig(max_length=args.max_tokens * 4)  # Approx chars per token
    summary = summarize_project(modules, config)
    
    print(summary)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Execute init command."""
    filename = f".code-analyzer.{args.format}"
    
    if Path(filename).exists():
        print(f"Config file already exists: {filename}")
        return 1
    
    config = ConfigManager()
    config.save(filename, args.format)
    
    print(f"Created config file: {filename}")
    return 0


def format_summary(summary: dict) -> str:
    """Format summary for display."""
    lines = [
        "=" * 50,
        "CODE ANALYSIS SUMMARY",
        "=" * 50,
        "",
    ]
    
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


def cmd_rag(args: argparse.Namespace) -> int:
    """Execute RAG commands."""
    if not hasattr(args, 'rag_command') or not args.rag_command:
        print("Usage: code-analyzer rag <command>")
        print("Commands: index, ask, search, clear, stats")
        return 1
    
    try:
        from analyzer.rag.pipeline import RAGPipeline
        from analyzer.rag.config import RAGConfig
        from analyzer.parsers import FileParser
    except ImportError as e:
        print(f"Error: RAG dependencies not available: {e}", file=sys.stderr)
        print("Install with: pip install chromadb openai", file=sys.stderr)
        return 1
    
    # Create RAG config with persist directory
    rag_config = RAGConfig()
    rag_config.vector_store.persist_directory = args.persist_dir
    
    if args.rag_command == "index":
        return cmd_rag_index(args, rag_config)
    elif args.rag_command == "ask":
        return cmd_rag_ask(args, rag_config)
    elif args.rag_command == "search":
        return cmd_rag_search(args, rag_config)
    elif args.rag_command == "clear":
        return cmd_rag_clear(args, rag_config)
    elif args.rag_command == "stats":
        return cmd_rag_stats(args, rag_config)
    else:
        print(f"Unknown RAG command: {args.rag_command}")
        return 1


def cmd_rag_index(args: argparse.Namespace, rag_config) -> int:
    """Index a codebase for RAG."""
    from analyzer.rag.pipeline import RAGPipeline
    from analyzer.parsers import FileParser
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        return 1
    
    print(f"Indexing: {path}")
    
    # Parse the codebase
    parser = FileParser()
    if path.is_file():
        modules = [parser.parse_file(path)]
    else:
        modules = parser.parse_directory(path, recursive=True)
    
    print(f"Parsed {len(modules)} modules")
    
    # Create pipeline and index
    pipeline = RAGPipeline(rag_config)
    stats = pipeline.index(modules, str(path), clear_existing=args.clear)
    
    print("\n" + "=" * 40)
    print("RAG INDEX COMPLETE")
    print("=" * 40)
    print(f"Total chunks indexed: {stats.total_chunks}")
    print(f"  - Modules: {stats.total_modules}")
    print(f"  - Classes: {stats.total_classes}")
    print(f"  - Functions: {stats.total_functions}")
    print(f"  - Methods: {stats.total_methods}")
    print(f"Persist directory: {stats.persist_directory}")
    print(f"Embedding provider: {stats.embedding_provider}")
    
    return 0


def cmd_rag_ask(args: argparse.Namespace, rag_config) -> int:
    """Ask a question about the indexed codebase."""
    from analyzer.rag.pipeline import RAGPipeline
    from analyzer.parsers import FileParser
    
    pipeline = RAGPipeline(rag_config)
    
    # Index path if provided
    if args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"Error: Path not found: {path}", file=sys.stderr)
            return 1
        
        print(f"Indexing: {path}")
        parser = FileParser()
        if path.is_file():
            modules = [parser.parse_file(path)]
        else:
            modules = parser.parse_directory(path, recursive=True)
        
        pipeline.index(modules, str(path))
    
    # Check if index exists
    if not pipeline.is_indexed():
        print("Error: No index found. Run 'code-analyzer rag index <path>' first.")
        return 1
    
    print(f"\nQuestion: {args.question}\n")
    print("-" * 40)
    
    # Query the codebase
    response = pipeline.query(args.question, top_k=args.top_k)
    
    print(response.answer)
    print("\n" + "-" * 40)
    print(response.format_sources())
    
    return 0


def cmd_rag_search(args: argparse.Namespace, rag_config) -> int:
    """Semantic search over the codebase."""
    from analyzer.rag.pipeline import RAGPipeline
    from analyzer.parsers import FileParser
    
    pipeline = RAGPipeline(rag_config)
    
    # Index path if provided
    if args.path:
        path = Path(args.path)
        if not path.exists():
            print(f"Error: Path not found: {path}", file=sys.stderr)
            return 1
        
        print(f"Indexing: {path}")
        parser = FileParser()
        if path.is_file():
            modules = [parser.parse_file(path)]
        else:
            modules = parser.parse_directory(path, recursive=True)
        
        pipeline.index(modules, str(path))
    
    # Check if index exists
    if not pipeline.is_indexed():
        print("Error: No index found. Run 'code-analyzer rag index <path>' first.")
        return 1
    
    print(f"Searching for: {args.query}\n")
    
    # Search
    filter_type = getattr(args, 'type', None)
    results = pipeline.search(args.query, top_k=args.top_k, filter_entity_type=filter_type)
    
    if not results:
        print("No results found.")
        return 0
    
    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        print(f"{i}. [{chunk.entity_type.upper()}] {chunk.full_name}")
        print(f"   File: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"   Score: {result.score:.3f}")
        # Show snippet
        snippet = chunk.content[:150].replace('\n', ' ')
        print(f"   {snippet}...")
        print()
    
    return 0


def cmd_rag_clear(args: argparse.Namespace, rag_config) -> int:
    """Clear the RAG index."""
    from analyzer.rag.pipeline import RAGPipeline
    
    pipeline = RAGPipeline(rag_config)
    pipeline.clear_index()
    
    print(f"Cleared RAG index from: {args.persist_dir}")
    return 0


def cmd_rag_stats(args: argparse.Namespace, rag_config) -> int:
    """Show RAG index statistics."""
    from analyzer.rag.pipeline import RAGPipeline
    
    pipeline = RAGPipeline(rag_config)
    stats = pipeline.get_stats()
    
    print("=" * 40)
    print("RAG INDEX STATISTICS")
    print("=" * 40)
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Persist directory: {rag_config.vector_store.persist_directory}")
    print(f"Embedding provider: {rag_config.embedding.provider}")
    print(f"LLM provider: {rag_config.llm.provider}")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == "analyze":
            return cmd_analyze(args)
        elif args.command == "query":
            return cmd_query(args)
        elif args.command == "summary":
            return cmd_summary(args)
        elif args.command == "init":
            return cmd_init(args)
        elif args.command == "rag":
            return cmd_rag(args)
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

