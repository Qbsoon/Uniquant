import os
import re
import subprocess

def find_imports_in_project(project_path):
    """
    Znajdź wszystkie importy w plikach .py w podanym katalogu.
    """
    imports = set()
    skip_dirs = {".venv", "venv", "__pycache__", ".git"}
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        # Szukaj linii zaczynających się od "import" lub "from"
                        match = re.match(r"^\s*(import|from)\s+([\w\.]+)", line)
                        if match:
                            imports.add(match.group(2).split(".")[0])  # Dodaj tylko główny moduł
    return imports

def generate_requirements_for_project(project_path, output_file="requirements.txt"):
    """
    Wygeneruj plik requirements.txt tylko dla używanych pakietów.
    """
    print("Szukam używanych pakietów...")
    used_imports = find_imports_in_project(project_path)
    print(f"Znalezione importy: {used_imports}")

    # Pobierz listę zainstalowanych pakietów
    installed_packages = {}
    freeze_output = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    for line in freeze_output:
        if "==" in line:  # Upewnij się, że linia zawiera nazwę i wersję pakietu
            pkg_name, pkg_version = line.split("==")
            installed_packages[pkg_name.lower()] = pkg_version

    # Dopasuj używane importy do zainstalowanych pakietów
    requirements = []
    for imp in used_imports:
        if imp.lower() in installed_packages:
            requirements.append(f"{imp}=={installed_packages[imp.lower()]}")

    # Zapisz do pliku requirements.txt
    with open(output_file, "w") as f:
        f.write("\n".join(requirements))
    print(f"Plik {output_file} został wygenerowany.")

if __name__ == "__main__":
    # Ścieżka do katalogu projektu
    project_path = os.path.dirname(os.path.abspath("main.py"))
    generate_requirements_for_project(project_path)