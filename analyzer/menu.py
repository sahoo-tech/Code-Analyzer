"""
Interactive CLI menu for the Code Analyzer.

Provides a user-friendly menu interface to access all analyzer features.
"""

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
        
        choice = input("Select option: ").strip()
        
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
