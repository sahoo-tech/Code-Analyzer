"""
Command-line interface for the Code Analyzer.
"""

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
