

import os
import sys
from pathlib import Path

# Try to import rich for enhanced output, fallback to basic if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Global RAG pipeline instance (shared across menu options)
_rag_pipeline = None

def get_rag_pipeline():
    """Get or create the shared RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        from analyzer.rag.pipeline import RAGPipeline
        from analyzer.rag.config import RAGConfig
        _rag_pipeline = RAGConfig()
        _rag_pipeline = RAGPipeline(_rag_pipeline)
    return _rag_pipeline

def reset_rag_pipeline():
    """Reset the RAG pipeline (used after clearing index)."""
    global _rag_pipeline
    _rag_pipeline = None



def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit(
            "[bold cyan]🔍 Enterprise Code Analyzer[/bold cyan]\n"
            "[dim]AI-Enhanced Static Analysis for Python[/dim]",
            border_style="cyan"
        ))
    else:
        print("=" * 50)
        print("🔍 ENTERPRISE CODE ANALYZER")
        print("   AI-Enhanced Static Analysis for Python")
        print("=" * 50)
    print()


def print_menu():
    """Print the main menu."""
    menu_items = [
        ("1", "📂 Analyze File", "Analyze a single Python file"),
        ("2", "📁 Analyze Directory", "Analyze entire project/module"),
        ("3", "📊 Quick Metrics", "Get complexity & LOC metrics"),
        ("4", "🔒 Security Scan", "Check for vulnerabilities & secrets"),
        ("5", "🎯 Pattern Detection", "Find design patterns & anti-patterns"),
        ("6", "🔗 Dependency Analysis", "Analyze imports & dependencies"),
        ("7", "🔍 Query Code", "Search code with natural language"),
        ("8", "📝 Generate Summary", "Get AI-friendly code summary"),
        ("A", "🤖 RAG AI Assistant", "Ask AI questions about your code"),
        ("9", "⚙️  Settings", "Configure analyzer options"),
        ("0", "❌ Exit", "Exit the analyzer"),
    ]
    
    if RICH_AVAILABLE:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold yellow", width=3)
        table.add_column("Feature", style="bold white", width=25)
        table.add_column("Description", style="dim")
        
        for key, feature, desc in menu_items:
            table.add_row(f"[{key}]", feature, desc)
        
        console = Console()
        console.print(table)
    else:
        for key, feature, desc in menu_items:
            print(f"  [{key}] {feature:25} - {desc}")
    
    print()


def get_path_input(prompt: str = "Enter path") -> str:
    """Get a file/directory path from user."""
    while True:
        path = input(f"\n{prompt}: ").strip()
        if not path:
            print("❌ Path cannot be empty. Try again.")
            continue
        if not Path(path).exists():
            print(f"❌ Path not found: {path}")
            retry = input("Try again? (y/n): ").lower()
            if retry != 'y':
                return ""
            continue
        return path


def display_results(result, title: str = "Analysis Results"):
    """Display analysis results."""
    summary = result.get_summary()
    
    print(f"\n{'=' * 50}")
    print(f"📊 {title}")
    print("=" * 50)
    
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {value}")
    
    print("=" * 50)
    
    input("\nPress Enter to continue...")


def analyze_file_menu():
    """Handle file analysis."""
    from analyzer import analyze_file
    
    print("\n📂 ANALYZE FILE")
    print("-" * 30)
    
    path = get_path_input("Enter file path")
    if not path:
        return
    
    print("\nAnalyzing... Please wait.")
    result = analyze_file(path)
    display_results(result, f"File Analysis: {Path(path).name}")
    
    # Offer to show details
    while True:
        print("\nShow details?")
        print("  [1] Security issues")
        print("  [2] Code smells")
        print("  [3] Patterns found")
        print("  [4] Full JSON")
        print("  [0] Back to menu")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            print("\n🔒 Security Issues:")
            for v in result.vulnerabilities[:10]:
                print(f"  - [{v.severity.value}] {v.message} (line {v.line_number})")
            for s in result.secrets[:10]:
                print(f"  - [secret] {s.message} (line {s.line_number})")
            if not result.vulnerabilities and not result.secrets:
                print("  ✅ No security issues found!")
            input("\nPress Enter...")
        elif choice == "2":
            print("\n🔍 Code Smells:")
            for smell in result.code_smells[:10]:
                print(f"  - [{smell.severity}] {smell.message} (line {smell.line_number})")
            if not result.code_smells:
                print("  ✅ No code smells found!")
            input("\nPress Enter...")
        elif choice == "3":
            print("\n🎯 Patterns Found:")
            print("  Design Patterns:")
            for p in result.design_patterns[:5]:
                print(f"    - {p.pattern_type.value}: {p.class_name}")
            print("  Anti-Patterns:")
            for p in result.anti_patterns[:5]:
                print(f"    - {p.pattern_type.value}: {p.entity_name}")
            if not result.design_patterns and not result.anti_patterns:
                print("  No patterns detected.")
            input("\nPress Enter...")
        elif choice == "4":
            print(result.to_json()[:2000], "..." if len(result.to_json()) > 2000 else "")
            input("\nPress Enter...")


def analyze_directory_menu():
    """Handle directory analysis."""
    from analyzer import analyze_directory
    
    print("\n📁 ANALYZE DIRECTORY")
    print("-" * 30)
    
    path = get_path_input("Enter directory path")
    if not path:
        return
    
    print("\nAnalyzing... This may take a moment.")
    result = analyze_directory(path)
    display_results(result, f"Project Analysis: {Path(path).name}")


def quick_metrics_menu():
    """Handle quick metrics."""
    from analyzer.parsers import FileParser
    from analyzer.metrics import calculate_complexity, calculate_loc, calculate_maintainability
    
    print("\n📊 QUICK METRICS")
    print("-" * 30)
    
    path = get_path_input("Enter file path")
    if not path:
        return
    
    code = Path(path).read_text(encoding='utf-8', errors='ignore')
    
    complexity = calculate_complexity(code)
    loc = calculate_loc(code)
    maintainability = calculate_maintainability(code)
    
    print(f"\n📊 Metrics for: {Path(path).name}")
    print("=" * 40)
    print(f"  Cyclomatic Complexity: {complexity.cyclomatic}")
    print(f"  Cognitive Complexity: {complexity.cognitive}")
    print(f"  Max Nesting Depth: {complexity.max_nesting_depth}")
    print()
    print(f"  Total Lines: {loc.total}")
    print(f"  Source Lines: {loc.source}")
    print(f"  Comment Lines: {loc.comments}")
    print(f"  Blank Lines: {loc.blank}")
    print()
    print(f"  Maintainability Index: {maintainability.maintainability_index:.1f}")
    print(f"  Rating: {maintainability.rating}")
    print("=" * 40)
    
    input("\nPress Enter to continue...")


def security_scan_menu():
    """Handle security scanning."""
    from analyzer import analyze_file, analyze_directory
    
    print("\n🔒 SECURITY SCAN")
    print("-" * 30)
    
    path = get_path_input("Enter file or directory path")
    if not path:
        return
    
    print("\nScanning for security issues...")
    
    if Path(path).is_file():
        result = analyze_file(path)
    else:
        result = analyze_directory(path)
    
    print(f"\n🔒 Security Report")
    print("=" * 40)
    
    if result.vulnerabilities:
        print("\n⚠️  VULNERABILITIES:")
        for v in result.vulnerabilities:
            print(f"  [{v.severity.value.upper()}] {v.message}")
            print(f"      File: {Path(v.file_path).name}:{v.line_number}")
            if v.recommendation:
                print(f"      Fix: {v.recommendation}")
            print()
    else:
        print("\n✅ No vulnerabilities found!")
    
    if result.secrets:
        print("\n🔑 HARDCODED SECRETS:")
        for s in result.secrets:
            print(f"  [{s.secret_type.value}] {s.message}")
            print(f"      File: {Path(s.file_path).name}:{s.line_number}")
            print()
    else:
        print("\n✅ No hardcoded secrets found!")
    
    print("=" * 40)
    input("\nPress Enter to continue...")


def pattern_detection_menu():
    """Handle pattern detection."""
    from analyzer import analyze_file, analyze_directory
    
    print("\n🎯 PATTERN DETECTION")
    print("-" * 30)
    
    path = get_path_input("Enter file or directory path")
    if not path:
        return
    
    print("\nDetecting patterns...")
    
    if Path(path).is_file():
        result = analyze_file(path)
    else:
        result = analyze_directory(path)
    
    print(f"\n🎯 Pattern Report")
    print("=" * 40)
    
    if result.design_patterns:
        print("\n✨ DESIGN PATTERNS:")
        for p in result.design_patterns:
            print(f"  {p.pattern_type.value}: {p.class_name}")
            print(f"      Confidence: {p.confidence:.0%}")
            for e in p.evidence[:2]:
                print(f"      - {e}")
            print()
    else:
        print("\n  No design patterns detected.")
    
    if result.anti_patterns:
        print("\n⚠️  ANTI-PATTERNS:")
        for p in result.anti_patterns:
            print(f"  [{p.severity}] {p.pattern_type.value}: {p.entity_name}")
            print(f"      {p.description}")
            print(f"      Suggestion: {p.suggestion}")
            print()
    
    print("=" * 40)
    input("\nPress Enter to continue...")


def dependency_analysis_menu():
    """Handle dependency analysis."""
    from analyzer.parsers import FileParser
    from analyzer.dependencies import analyze_imports, build_module_graph
    
    print("\n🔗 DEPENDENCY ANALYSIS")
    print("-" * 30)
    
    path = get_path_input("Enter directory path")
    if not path:
        return
    
    print("\nAnalyzing dependencies...")
    
    parser = FileParser()
    modules = parser.parse_directory(path)
    
    print(f"\n🔗 Dependency Report")
    print("=" * 40)
    
    print(f"\nModules analyzed: {len(modules)}")
    
    # Analyze imports
    stdlib = set()
    third_party = set()
    local = set()
    
    for module in modules:
        analysis = analyze_imports(module, Path(path))
        stdlib.update(analysis.stdlib_imports)
        third_party.update(analysis.third_party_imports)
        local.update(analysis.local_imports)
    
    print(f"\n📦 Standard Library: {len(stdlib)}")
    for imp in list(stdlib)[:5]:
        print(f"    - {imp}")
    if len(stdlib) > 5:
        print(f"    ... and {len(stdlib) - 5} more")
    
    print(f"\n📚 Third-Party: {len(third_party)}")
    for imp in list(third_party)[:5]:
        print(f"    - {imp}")
    if len(third_party) > 5:
        print(f"    ... and {len(third_party) - 5} more")
    
    print(f"\n🏠 Local Imports: {len(local)}")
    
    # Check for circular dependencies
    graph = build_module_graph(modules, Path(path))
    cycles = graph.find_circular_dependencies()
    
    if cycles:
        print(f"\n⚠️  Circular Dependencies: {len(cycles)}")
        for cycle in cycles[:3]:
            print(f"    {' -> '.join(cycle)}")
    else:
        print("\n✅ No circular dependencies!")
    
    print("=" * 40)
    input("\nPress Enter to continue...")


def query_code_menu():
    """Handle code querying."""
    from analyzer import analyze_directory
    from analyzer.ai import QueryInterface
    
    print("\n🔍 QUERY CODE")
    print("-" * 30)
    
    path = get_path_input("Enter directory path")
    if not path:
        return
    
    print("\nLoading codebase...")
    result = analyze_directory(path)
    interface = QueryInterface(result.modules)
    
    print("\n✅ Ready! Enter queries or 'exit' to quit.")
    print("Examples:")
    print("  - find function main")
    print("  - find class Config")
    print("  - list all async functions")
    print("  - find classes inheriting from ABC")
    print()
    
    while True:
        query = input("Query> ").strip()
        
        if query.lower() in ('exit', 'quit', 'q'):
            break
        
        if not query:
            continue
        
        result = interface.query(query)
        
        print(f"\nFound {result.count} results:")
        for r in result.results[:10]:
            if isinstance(r, dict):
                print(f"  - {r.get('type', 'item')}: {r.get('name', str(r))}")
            else:
                print(f"  - {r}")
        
        if result.count > 10:
            print(f"  ... and {result.count - 10} more")
        print()


def generate_summary_menu():
    """Handle summary generation."""
    from analyzer import analyze_directory
    from analyzer.ai import format_for_ai
    
    print("\n📝 GENERATE SUMMARY")
    print("-" * 30)
    
    path = get_path_input("Enter directory path")
    if not path:
        return
    
    print("\nGenerating summary...")
    result = analyze_directory(path)
    
    print("\nOutput format?")
    print("  [1] Markdown (human-readable)")
    print("  [2] JSON (programmatic)")
    
    choice = input("\nChoice: ").strip()
    format_type = "markdown" if choice == "1" else "json"
    
    from analyzer import CodeAnalyzer
    analyzer = CodeAnalyzer()
    summary = analyzer.get_ai_summary(result, format_type)
    
    print(f"\n{summary[:3000]}")
    if len(summary) > 3000:
        print("\n... (truncated)")
    
    # Offer to save
    save = input("\nSave to file? (y/n): ").lower()
    if save == 'y':
        ext = "md" if format_type == "markdown" else "json"
        filename = f"analysis_summary.{ext}"
        Path(filename).write_text(summary, encoding='utf-8')
        print(f"✅ Saved to {filename}")
    
    input("\nPress Enter to continue...")


def rag_menu():
    """Handle RAG AI Assistant menu."""
    print("\n🤖 RAG AI ASSISTANT")
    print("-" * 30)
    print("\nThis feature uses AI to answer questions about your code.")
    print("It can search semantically and understand code context.")
    
    # Check for AI availability
    import os
    has_ai = any([
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
    ])
    
    if has_ai:
        print("\n✅ AI provider detected!")
    else:
        print("\n⚠️  No AI API key found. Using demo mode.")
        print("   Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY for real AI.")
    
    while True:
        print("\n🤖 RAG Options:")
        print("  [1] 📦 Index Codebase - Index a project for AI search")
        print("  [2] 💬 Ask Question - Ask AI about your code")
        print("  [3] 🔎 Semantic Search - Find similar code")
        print("  [4] 🎮 Demo Mode - Try with sample data")
        print("  [5] 📊 Index Stats - View current index info")
        print("  [6] 🗑️  Clear Index - Remove indexed data")
        print("  [0] ⬅️  Back to Main Menu")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            rag_index_menu()
        elif choice == "2":
            rag_ask_menu()
        elif choice == "3":
            rag_search_menu()
        elif choice == "4":
            rag_demo_menu()
        elif choice == "5":
            rag_stats_menu()
        elif choice == "6":
            rag_clear_menu()
        else:
            print("❌ Invalid option.")


def rag_index_menu():
    """Index a codebase for RAG."""
    print("\n📦 INDEX CODEBASE")
    print("-" * 30)
    
    path = get_path_input("Enter directory path to index")
    if not path:
        return
    
    try:
        from analyzer.parsers import FileParser
        
        print("\nParsing code...")
        parser = FileParser()
        modules = parser.parse_directory(path, recursive=True)
        print(f"Found {len(modules)} Python modules")
        
        print("\nIndexing for RAG (this may take a moment)...")
        pipeline = get_rag_pipeline()
        stats = pipeline.index(modules, str(path), clear_existing=True)
        
        print("\n" + "=" * 40)
        print("✅ INDEXING COMPLETE")
        print("=" * 40)
        print(f"  Total chunks: {stats.total_chunks}")
        print(f"  - Modules: {stats.total_modules}")
        print(f"  - Classes: {stats.total_classes}")
        print(f"  - Functions: {stats.total_functions}")
        print(f"  - Methods: {stats.total_methods}")
        print("=" * 40)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")


def rag_ask_menu():
    """Ask a question about the indexed code."""
    print("\n💬 ASK AI ABOUT YOUR CODE")
    print("-" * 30)
    
    try:
        pipeline = get_rag_pipeline()
        
        if not pipeline.is_indexed():
            print("\n⚠️  No code indexed yet!")
            print("   Use option [1] to index a codebase first,")
            print("   or option [4] to try demo mode.")
            input("\nPress Enter to continue...")
            return
        
        print("\nReady! Type your question or 'exit' to quit.")
        print("Examples:")
        print("  - How does the Calculator class work?")
        print("  - What authentication methods are available?")
        print("  - Find error handling patterns")
        print()
        
        while True:
            question = input("Question> ").strip()
            
            if question.lower() in ('exit', 'quit', 'q', ''):
                break
            
            print("\n🔍 Searching and analyzing...")
            response = pipeline.query(question)
            
            print("\n" + "=" * 50)
            print("🤖 AI Response:")
            print("=" * 50)
            print(response.answer)
            print("\n" + "-" * 50)
            print(response.format_sources())
            print()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\nPress Enter to continue...")


def rag_search_menu():
    """Semantic search over code."""
    print("\n🔎 SEMANTIC SEARCH")
    print("-" * 30)
    
    try:
        pipeline = get_rag_pipeline()
        
        if not pipeline.is_indexed():
            print("\n⚠️  No code indexed yet!")
            input("\nPress Enter to continue...")
            return
        
        query = input("\nSearch query: ").strip()
        if not query:
            return
        
        print("\n🔍 Searching...")
        results = pipeline.search(query, top_k=10)
        
        if not results:
            print("\nNo results found.")
        else:
            print(f"\nFound {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                chunk = result.chunk
                print(f"{i}. [{chunk.entity_type.upper()}] {chunk.full_name}")
                print(f"   File: {chunk.file_path}:{chunk.start_line}")
                print(f"   Score: {result.score:.3f}")
                snippet = chunk.content[:100].replace('\n', ' ')
                print(f"   {snippet}...")
                print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")


def rag_demo_menu():
    """Run RAG demo with sample data."""
    print("\n🎮 RAG DEMO MODE")
    print("-" * 30)
    print("\nThis will create sample code and demonstrate RAG capabilities.")
    
    confirm = input("\nProceed? (y/n): ").lower()
    if confirm != 'y':
        return
    
    try:
        from analyzer.rag.demo_data import create_demo_project, get_demo_questions
        from analyzer.parsers import FileParser
        
        print("\n1️⃣  Creating demo project...")
        demo_path = create_demo_project()
        print(f"   Created demo files in: {demo_path}")
        
        print("\n2️⃣  Parsing demo code...")
        parser = FileParser()
        modules = parser.parse_directory(str(demo_path), recursive=True)
        print(f"   Found {len(modules)} modules")
        
        print("\n3️⃣  Indexing for RAG...")
        pipeline = get_rag_pipeline()
        stats = pipeline.index(modules, str(demo_path), clear_existing=True)
        print(f"   Indexed {stats.total_chunks} code chunks")
        
        print("\n✅ Demo ready!")
        print("\n" + "=" * 50)
        
        # Show demo questions
        demo_questions = get_demo_questions()
        print("\n📋 Sample questions you can ask:")
        for i, q in enumerate(demo_questions[:5], 1):
            print(f"   [{i}] {q}")
        print("   [c] Custom question")
        print("   [0] Exit demo")
        
        while True:
            choice = input("\nSelect a question or type 'c' for custom: ").strip()
            
            if choice == '0':
                break
            elif choice == 'c':
                question = input("Your question: ").strip()
                if not question:
                    continue
            elif choice.isdigit() and 1 <= int(choice) <= len(demo_questions):
                question = demo_questions[int(choice) - 1]
                print(f"\n❓ {question}")
            else:
                print("Invalid choice.")
                continue
            
            print("\n🔍 Searching and analyzing...")
            response = pipeline.query(question)
            
            print("\n" + "=" * 50)
            print("🤖 AI Response:")
            print("=" * 50)
            print(response.answer)
            print("\n" + "-" * 50)
            print(response.format_sources())
            print()
        
        # Cleanup option
        cleanup = input("\nRemove demo files? (y/n): ").lower()
        if cleanup == 'y':
            from analyzer.rag.demo_data import cleanup_demo_project
            cleanup_demo_project(demo_path)
            print("Demo files removed.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")


def rag_stats_menu():
    """Show RAG index statistics."""
    print("\n📊 RAG INDEX STATS")
    print("-" * 30)
    
    try:
        import os
        
        pipeline = get_rag_pipeline()
        config = pipeline.config
        stats = pipeline.get_stats()
        
        print("\n" + "=" * 40)
        print("📊 Current RAG Index")
        print("=" * 40)
        print(f"  Total chunks indexed: {stats.total_chunks}")
        print(f"  Persist directory: {config.vector_store.persist_directory}")
        print(f"  Embedding provider: {config.embedding.provider}")
        print(f"  LLM provider: {config.llm.provider}")
        
        # Check AI availability
        print("\n🔌 AI Providers Available:")
        providers = [
            ("GEMINI_API_KEY", "Google Gemini"),
            ("GOOGLE_API_KEY", "Google Gemini"),
            ("OPENAI_API_KEY", "OpenAI GPT"),
            ("ANTHROPIC_API_KEY", "Anthropic Claude"),
        ]
        for env_var, name in providers:
            status = "✅" if os.getenv(env_var) else "❌"
            print(f"  {status} {name} ({env_var})")
        
        print("=" * 40)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")


def rag_clear_menu():
    """Clear the RAG index."""
    print("\n🗑️  CLEAR RAG INDEX")
    print("-" * 30)
    
    confirm = input("\nAre you sure you want to clear the index? (yes/no): ").lower()
    if confirm != 'yes':
        print("Cancelled.")
        input("\nPress Enter to continue...")
        return
    
    try:
        pipeline = get_rag_pipeline()
        pipeline.clear_index()
        reset_rag_pipeline()  # Reset to create fresh instance next time
        
        print("\n✅ RAG index cleared.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    input("\nPress Enter to continue...")


def settings_menu():
    """Handle settings."""
    print("\n⚙️  SETTINGS")
    print("-" * 30)
    print("\nCurrent settings are loaded from:")
    print("  - .code-analyzer.yaml")
    print("  - .code-analyzer.json")
    print("  - Environment variables (CODE_ANALYZER_*)")
    print()
    print("Run 'code-analyzer init' to create a config file.")
    input("\nPress Enter to continue...")


def main():
    """Main menu loop."""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("Select option: ").strip().upper()
        
        try:
            if choice == "0":
                print("\n👋 Goodbye!")
                sys.exit(0)
            elif choice == "1":
                analyze_file_menu()
            elif choice == "2":
                analyze_directory_menu()
            elif choice == "3":
                quick_metrics_menu()
            elif choice == "4":
                security_scan_menu()
            elif choice == "5":
                pattern_detection_menu()
            elif choice == "6":
                dependency_analysis_menu()
            elif choice == "7":
                query_code_menu()
            elif choice == "8":
                generate_summary_menu()
            elif choice == "A":
                rag_menu()
            elif choice == "9":
                settings_menu()
            else:
                print("\n❌ Invalid option. Please try again.")
                input("Press Enter...")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

