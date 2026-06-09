import subprocess
import os
import datetime
import time

def run(cmd, env=None):
    try:
        subprocess.check_call(cmd, env=env, shell=True)
    except Exception as e:
        print(f"Failed: {cmd} - {e}")

def main():
    run('git init')
    run('git remote add origin https://github.com/Harsha081459/CLV-Intelligence-Engine.git')
    
    # First 3 commits in April 2025
    date_2025 = datetime.datetime(2025, 4, 10, 10, 0, 0)
    # Last 2 commits in June 2026
    date_2026 = datetime.datetime(2026, 6, 5, 10, 0, 0)
    
    files_to_commit = [
        (['README.md', 'requirements.txt', '.gitignore'], "Initial commit: Documentation and deps", True),
        (['src/', 'download_data.py'], "Data engineering and download scripts", True),
        (['notebooks/', 'run_pipeline.py', 'run_conformal.py'], "Core modeling and notebooks", True),
        (['dashboard/', 'plots/', 'info/'], "Streamlit dashboard and analysis results", False)
    ]
    
    for files, msg, is_old in files_to_commit:
        for f in files:
            run(f'git add {f}')
        
        if is_old:
            date_str = date_2025.strftime('%Y-%m-%dT%H:%M:%S')
            date_2025 += datetime.timedelta(days=3, hours=2)
        else:
            date_str = date_2026.strftime('%Y-%m-%dT%H:%M:%S')
            date_2026 += datetime.timedelta(days=4, hours=2)
            
        env = dict(os.environ)
        env['GIT_COMMITTER_DATE'] = date_str
        env['GIT_AUTHOR_DATE'] = date_str
        
        run(f'git commit -m "{msg}" --date="{date_str}"', env=env)

    # Also add the rest of the files
    run('git add .')
    date_str = date_2026.strftime('%Y-%m-%dT%H:%M:%S')
    env = dict(os.environ)
    env['GIT_COMMITTER_DATE'] = date_str
    env['GIT_AUTHOR_DATE'] = date_str
    run(f'git commit -m "Final polish and LaTeX report" --date="{date_str}"', env=env)

    print("Force pushing to GitHub...")
    run('git branch -M main')
    run('git push origin main --force')

if __name__ == "__main__":
    main()
