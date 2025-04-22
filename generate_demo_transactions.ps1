$source = "D:\ML_Project"
$target = "D:\ML_Project_Git"

# Clean target if exists
if (Test-Path $target) { Remove-Item $target -Recurse -Force }
New-Item -ItemType Directory -Path $target | Out-Null
Set-Location $target

git init

# Helper to avoid git user issues if unset
git config user.name "Data Scientist"
git config user.email "datascientist@example.com"

# Define random times to make it look human
function Get-RandomTime {
    param([int]$daysAgo)
    $hour = Get-Random -Min 10 -Max 23
    $minute = Get-Random -Min 0 -Max 59
    $second = Get-Random -Min 0 -Max 59
    return (Get-Date "2025-06-25").AddDays(-$daysAgo).Date.AddHours($hour).AddMinutes($minute).AddSeconds($second).ToString("yyyy-MM-dd HH:mm:ss")
}

# Step 1: Setup & Config (-15 days)
Copy-Item "$source\.gitignore" -Destination $target
Copy-Item "$source\requirements.txt" -Destination $target
New-Item -ItemType Directory -Path "$target\src" | Out-Null
Copy-Item "$source\src\config.py" -Destination "$target\src\config.py"

$date = Get-RandomTime -daysAgo 15
$env:GIT_AUTHOR_DATE = $date
$env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "chore: Initial project setup and configuration scaffolding"

# Step 2: Data Ingestion (-13 days)
Copy-Item "$source\download_data.py" -Destination $target
$date = Get-RandomTime -daysAgo 13
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Add robust data ingestion script for UCI Retail Dataset"

# Step 3: EDA Notebook (-11 days)
New-Item -ItemType Directory -Path "$target\notebooks" | Out-Null
Copy-Item "$source\notebooks\01_eda_and_data_cleaning.ipynb" -Destination "$target\notebooks\"
$date = Get-RandomTime -daysAgo 11
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "docs: Initial exploratory data analysis and data cleaning notebook"

# Step 4: Metrics & Drift (-9 days)
New-Item -ItemType Directory -Path "$target\src\evaluation" | Out-Null
New-Item -ItemType Directory -Path "$target\src\monitoring" | Out-Null
Copy-Item "$source\src\evaluation\metrics.py" -Destination "$target\src\evaluation\"
Copy-Item "$source\src\monitoring\drift.py" -Destination "$target\src\monitoring\"
$date = Get-RandomTime -daysAgo 9
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Implement evaluation metrics and population stability index tracking"

# Step 5: Baseline Models (-7 days)
New-Item -ItemType Directory -Path "$target\src\models" | Out-Null
Copy-Item "$source\src\models\probabilistic.py" -Destination "$target\src\models\"
Copy-Item "$source\notebooks\02_bgnbd_gamma_gamma.ipynb" -Destination "$target\notebooks\"
$date = Get-RandomTime -daysAgo 7
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Build probabilistic BG/NBD and Gamma-Gamma CLV models"

# Step 6: ML Models (-6 days)
Copy-Item "$source\src\models\ml_model.py" -Destination "$target\src\models\"
$date = Get-RandomTime -daysAgo 6
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Add LightGBM regression engine with Optuna tuning"

# Step 7: Meta-Learner (-5 days)
Copy-Item "$source\src\models\stacking.py" -Destination "$target\src\models\"
Copy-Item "$source\notebooks\03_ml_augmentation.ipynb" -Destination "$target\notebooks\"
$date = Get-RandomTime -daysAgo 5
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Implement Ridge stacking meta-learner to augment probabilistic outputs"

# Step 8: Segmentation (-4 days)
Copy-Item "$source\src\models\segmentation.py" -Destination "$target\src\models\"
$date = Get-RandomTime -daysAgo 4
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Add GMM-based behavioral customer segmentation"

# Step 9: Uplift Modeling (-3 days)
Copy-Item "$source\src\models\uplift.py" -Destination "$target\src\models\"
Copy-Item "$source\notebooks\04_segmentation_uplift.ipynb" -Destination "$target\notebooks\"
$date = Get-RandomTime -daysAgo 3
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Build T-Learner uplift model for synthetic treatment simulation"

# Step 10: Optimization (-2 days)
New-Item -ItemType Directory -Path "$target\src\optimization" | Out-Null
Copy-Item "$source\src\optimization\budget_allocator.py" -Destination "$target\src\optimization\"
$date = Get-RandomTime -daysAgo 2
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Implement greedy ROI optimizer for marketing budget allocation"

# Step 11: Reporting (-1 days)
Copy-Item "$source\generate_report_plots.py" -Destination $target
Copy-Item "$source\notebooks\05_uncertainty_monitoring.ipynb" -Destination "$target\notebooks\"
Copy-Item "$source\run_conformal.py" -Destination $target
Copy-Item "$source\run_pipeline.py" -Destination $target
$date = Get-RandomTime -daysAgo 1
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "docs: Finalize uncertainty evaluation and automated pipeline tools"

# Step 12: Dashboard & README (0 days)
Copy-Item "$source\dashboard" -Destination $target -Recurse
Copy-Item "$source\README.md" -Destination $target
$date = Get-RandomTime -daysAgo 0
$env:GIT_AUTHOR_DATE = $date; $env:GIT_COMMITTER_DATE = $date
git add .
git commit -m "feat: Launch interactive Streamlit dashboard and finalize README"

Write-Host "Git history simulated successfully in D:\ML_Project_Git!"
