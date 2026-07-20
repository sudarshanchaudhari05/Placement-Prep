@echo off
cd /d "C:\Path\To\Your\Placement-Preparation-Folder"
git add .
git commit -m "Auto-update study materials: %date% %time%"
git push origin main