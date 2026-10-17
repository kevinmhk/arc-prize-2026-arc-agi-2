import kagglehub

# Download latest version
path = kagglehub.competition_download("arc-prize-2026-arc-agi-2")

print("Path to competition files:", path)
