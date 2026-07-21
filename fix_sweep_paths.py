import os, glob, shutil
ROOT = os.path.expanduser("~/Desktop/EEG_Image_decode-develop")
OLD = "cd ~/Desktop/EEG_Image_decode-develop"
NEW = 'REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\ncd "$REPO"'
changed = 0
for sh in glob.glob(f"{ROOT}/*.sh"):
    src = open(sh, encoding="utf-8", errors="ignore").read()
    if OLD in src:
        bak = sh + ".bak_repopath"
        if not os.path.exists(bak): shutil.copy(sh, bak)
        src = src.replace(OLD, NEW, 1)
        open(sh, "w", encoding="utf-8", newline="\n").write(src)
        print(f"fixed + LF endings: {os.path.basename(sh)}"); changed += 1
print(f"\n{changed} scripts updated (location-independent + Linux line endings)")
