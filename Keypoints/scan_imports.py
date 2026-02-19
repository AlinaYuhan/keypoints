import os
import re
import glob

def extract_imports(file_path):
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        for line in content:
            line = line.strip()
            if line.startswith('import ') and not line.startswith('#'):
                match = re.match(r'import\s+([^\s#]+)', line)
                if match:
                    imports.add(match.group(1).split('.')[0])
            elif line.startswith('from ') and not line.startswith('#'):
                match = re.match(r'from\s+([^\s#]+)\s+import', line)
                if match:
                    imports.add(match.group(1).split('.')[0])
    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    return imports

def main():
    all_imports = set()
    
    for py_file in glob.glob("**/*.py", recursive=True):
        print(f"扫描文件: {py_file}")
        file_imports = extract_imports(py_file)
        all_imports.update(file_imports)
    
    standard_libs = {
        'os', 'sys', 'json', 'math', 're', 'glob', 'argparse', 
        'collections', 'time', 'datetime', 'functools', 'itertools'
    }
    
    third_party_imports = all_imports - standard_libs
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        for lib in sorted(third_party_imports):
            f.write(lib + '\n')
    
    print(f"\n找到 {len(third_party_imports)} 个第三方库:")
    for lib in sorted(third_party_imports):
        print(f"  {lib}")
    
    print("\n已生成 requirements.txt 文件")

if __name__ == "__main__":
    main()



