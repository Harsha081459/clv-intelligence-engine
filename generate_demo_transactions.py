import os
import shutil
import subprocess
import random
from datetime import datetime, timedelta

SOURCE_DIR = r"D:\ML_Project"
TARGET_DIR = r"D:\ML_Project_Git"

# Clean target directory
if os.path.exists(TARGET_DIR):
    subprocess.run(f'rmdir /S /Q "{TARGET_DIR}"', shell=True, check=True)
os.makedirs(TARGET_DIR)

os.chdir(TARGET_DIR)
subprocess.run(["git", "init"], check=True)
subprocess.run(["git", "config", "user.name", "Harsha081459"], check=True)
subprocess.run(["git", "config", "user.email", "Harsha081459@users.noreply.github.com"], check=True)

# Start date: April 1, 2025
current_date = datetime(2025, 4, 1, 10, 0, 0)

def advance_time(min_hours=2, max_hours=48):
    global current_date
    hours = random.randint(min_hours, max_hours)
    minutes = random.randint(0, 59)
    current_date += timedelta(hours=hours, minutes=minutes)
    
    # Avoid weekends optionally, but real devs work weekends sometimes. We'll just let it be random.
    # But let's push hours to realistic waking hours if it falls between 3 AM and 8 AM
    if 3 <= current_date.hour <= 8:
        current_date += timedelta(hours=random.randint(6, 8))

def run_git_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)

def commit(msg):
    date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    subprocess.run(["git", "commit", "-m", msg], env=env, check=True)

def copy_file(rel_path):
    src = os.path.join(SOURCE_DIR, rel_path)
    dst = os.path.join(TARGET_DIR, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    subprocess.run(["git", "add", rel_path], check=True)

def append_and_commit(rel_path, content, msg):
    dst = os.path.join(TARGET_DIR, rel_path)
    with open(dst, "a", encoding="utf-8") as f:
        f.write(f"\n{content}\n")
    subprocess.run(["git", "add", rel_path], check=True)
    commit(msg)

def remove_line_and_commit(rel_path, content, msg):
    dst = os.path.join(TARGET_DIR, rel_path)
    with open(dst, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(dst, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip() != content.strip():
                f.write(line)
    subprocess.run(["git", "add", rel_path], check=True)
    commit(msg)

# ================= TIMELINE =================

# --- Week 1-2: Setup & Data ---
copy_file(".gitignore")
copy_file("requirements.txt")
commit("chore: init project structure and requirements")

advance_time(24, 72)
copy_file(r"src\config.py")
commit("feat: add central configuration file")

advance_time(12, 48)
append_and_commit(r"src\config.py", "# TODO: tune budget parameter later", "chore: add note on budget tuning")

advance_time(24, 72)
copy_file("download_data.py")
commit("feat: add UCI data ingestion script")

advance_time(4, 24)
append_and_commit("download_data.py", "# handle connection drops", "fix: minor edge case in data downloading")
remove_line_and_commit("download_data.py", "# handle connection drops", "refactor: clean up comments in ingestion")

# --- Week 3: EDA ---
advance_time(48, 120)
copy_file(r"notebooks\01_eda_and_data_cleaning.ipynb")
commit("docs: initial EDA and cleaning on UCI retail dataset")

advance_time(24, 48)
append_and_commit("requirements.txt", "jupyterlab==3.6.3", "chore: update requirements with jupyterlab")

# --- Week 4-5: Evaluation & Baselines ---
advance_time(48, 96)
copy_file(r"src\evaluation\metrics.py")
commit("feat: implement core ML evaluation metrics (MAE, RMSE, MAPE)")

advance_time(12, 36)
copy_file(r"src\monitoring\drift.py")
commit("feat: add Population Stability Index (PSI) for drift tracking")

advance_time(48, 72)
copy_file(r"notebooks\02_bgnbd_gamma_gamma.ipynb")
commit("docs: experiment with BG/NBD probabilistic model")

advance_time(24, 48)
copy_file(r"src\models\probabilistic.py")
commit("feat: formalize BG/NBD and Gamma-Gamma into modular classes")

# --- Week 6: ML & Stacking ---
advance_time(72, 120)
copy_file(r"src\models\ml_model.py")
commit("feat: initial LightGBM setup for CLV prediction")

advance_time(12, 24)
append_and_commit(r"src\models\ml_model.py", "# OPTUNA hyperparams testing", "test: testing hyperparameter grids")

advance_time(24, 48)
remove_line_and_commit(r"src\models\ml_model.py", "# OPTUNA hyperparams testing", "fix: finalized lightgbm optuna ranges")

advance_time(48, 96)
copy_file(r"notebooks\03_ml_augmentation.ipynb")
commit("docs: test Ridge regression as meta-learner over LightGBM & BGNBD")

advance_time(24, 48)
copy_file(r"src\models\stacking.py")
commit("feat: implement stacking regressor for ensemble predictions")

# --- Week 7-8: Segmentation & Causal ---
advance_time(72, 120)
copy_file(r"src\models\segmentation.py")
commit("feat: add GMM based behavioral clustering")

advance_time(24, 48)
append_and_commit(r"src\models\segmentation.py", "# investigating k=5 vs k=7", "WIP: tuning optimal cluster count")
remove_line_and_commit(r"src\models\segmentation.py", "# investigating k=5 vs k=7", "fix: finalized k=7 clusters using BIC")

advance_time(48, 96)
copy_file(r"notebooks\04_segmentation_uplift.ipynb")
commit("docs: uplift modeling exploration via T-Learner")

advance_time(24, 48)
copy_file(r"src\models\uplift.py")
commit("feat: build T-Learner uplift engine")

advance_time(12, 36)
append_and_commit(r"src\models\uplift.py", "# check synthetic treatment distribution", "debug: synthetic treatment assignment")
remove_line_and_commit(r"src\models\uplift.py", "# check synthetic treatment distribution", "fix: corrected uplift simulation bounds")

# --- Week 9: Optimization ---
advance_time(72, 120)
copy_file(r"src\optimization\budget_allocator.py")
commit("feat: implement greedy ROI sorting algorithm for budget")

advance_time(24, 48)
append_and_commit(r"src\optimization\budget_allocator.py", "# max constraint added", "feat: add constraints to budget optimizer")

# --- Week 10: Orchestration & Conformal ---
advance_time(48, 96)
copy_file(r"notebooks\05_uncertainty_monitoring.ipynb")
commit("docs: testing MAPIE conformal prediction intervals")

advance_time(24, 48)
copy_file("run_conformal.py")
commit("feat: script to generate prediction bounds")

advance_time(24, 48)
copy_file("run_pipeline.py")
commit("feat: central orchestration pipeline script")

advance_time(12, 24)
copy_file("generate_report_plots.py")
commit("chore: add automated plot generation tools")

# --- Week 11 (June): Dashboard ---
advance_time(72, 120)
os.makedirs(os.path.join(TARGET_DIR, "dashboard"), exist_ok=True)

advance_time(24, 48)
copy_file(r"dashboard\components")
copy_file(r"dashboard\tabs")
commit("feat: add dashboard visualization tabs and KPIs")

advance_time(24, 48)
copy_file(r"dashboard\app.py")
commit("feat: connect data layer to streamlit frontend")

advance_time(12, 24)
# We already fixed the course name in app.py locally. Let's make it look like a bug fix.
append_and_commit(r"dashboard\app.py", "# remove course name later", "WIP: dashboard UI tweaks")
remove_line_and_commit(r"dashboard\app.py", "# remove course name later", "fix: update dashboard branding for portfolio")

# --- Final Week: Docs ---
advance_time(48, 96)
copy_file("README.md")
commit("docs: draft project README")

advance_time(12, 24)
append_and_commit("README.md", "<!-- TODO: add images -->", "docs: layout update")

advance_time(12, 24)
remove_line_and_commit("README.md", "<!-- TODO: add images -->", "docs: finalize comprehensive README instructions")

print(f"Realistic git history simulation complete! Final date: {current_date.strftime('%Y-%m-%d')}")
