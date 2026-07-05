#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

KERNELSU_REPOS=(
    "https://github.com/KernelSU-Next/KernelSU-Next"
    "https://github.com/ReSukiSU/ReSukiSU"
    "https://github.com/rsuntk/KernelSU"
    "https://github.com/backslashxx/KernelSU/"
)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

get_repo_branch() {
    local repo_url=$1
    repo_url="${repo_url%/}"
    case "$repo_url" in
        *KernelSU-Next*)        echo "legacy"    ;;
        *resukisu*)         echo "builtin"   ;;
        *rsuntk/KernelSU*)       echo "main"   ;;
        *backslashxx/KernelSU*)  echo "master" ;;
        *)                       echo "main"   ;;
    esac
}

get_local_folder_name() {
    local repo_url=$1
    repo_url="${repo_url%/}"
    case "$repo_url" in
        *KernelSU-Next*)        echo "ksunext"     ;;
        *resukisu*)         echo "resukisu" ;;
        *rsuntk/KernelSU*)       echo "rksu"        ;;
        *backslashxx/KernelSU*)  echo "xxksu"       ;;
        *)                       basename "$repo_url" .git | tr '[:upper:]' '[:lower:]' ;;
    esac
}

get_remote_hash() {
    local repo_url=$1
    local branch=$2
    git ls-remote "$repo_url" "refs/heads/$branch" | awk '{print $1}'
}

do_sparse_clone() {
    local repo_url=$1
    local dest_dir=$2
    local branch=$3
    local remote_hash=$4

    echo -e "${BLUE}[*] Initializing sparse clone for:${NC} $repo_url ($branch)"
    
    mkdir -p "$dest_dir"
    cd "$dest_dir" || exit 1
    git init -q
    
    git config core.sparseCheckout true
    echo "kernel/" > .git/info/sparse-checkout
    
    git remote add origin "$repo_url"
    echo -e "${BLUE}[*] Pulling kernel folder...${NC}"
    git pull --depth=1 origin "$branch" -q

    if [ -d "kernel" ]; then
        echo "$remote_hash" > .version_hash
        rm -rf .git .gitattributes 2>/dev/null
        echo -e "${GREEN}[✓] Successfully pulled kernel folder into $(basename "$dest_dir")${NC}\n"
    else
        echo -e "${RED}[✗] Failed to fetch kernel folder.${NC}\n"
        rm -rf .git
    fi
    cd "$SCRIPT_DIR" || exit
}

# Main Execution Flow
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}        KernelSU Direct Tracker & Updater      ${NC}"
echo -e "${BLUE}===============================================${NC}\n"

if [ ${#KERNELSU_REPOS[@]} -eq 0 ]; then
    echo -e "${RED}[!] No repositories registered.${NC}"
    exit 1
fi

cd "$SCRIPT_DIR" || exit 1

for repo in "${KERNELSU_REPOS[@]}"; do
    folder_name=$(get_local_folder_name "$repo")
    target_branch=$(get_repo_branch "$repo")
    local_repo_path="${SCRIPT_DIR}/${folder_name}"
    
    echo -e "${BLUE}[+] Checking repo:${NC} $repo"

    remote_hash=$(get_remote_hash "$repo" "$target_branch")
    if [ -z "$remote_hash" ]; then
        echo -e "${RED}[✗] Could not reach remote repository. Skipping.${NC}\n"
        continue
    fi

    if [ -f "$local_repo_path/.version_hash" ]; then
        local_hash=$(cat "$local_repo_path/.version_hash")

        echo -e "${YELLOW}[i] Local hash:  ${local_hash:0:7}${NC}"
        echo -e "${YELLOW}[i] Remote hash: ${remote_hash:0:7}${NC}"

        if [ "$local_hash" == "$remote_hash" ]; then
            echo -e "${GREEN}[✓] Already up-to-date.${NC}\n"
        else
            echo -e "${GREEN}[!] Update available!${NC}"
            read -p "Do you want to pull the updated kernel folder? (y/N): " -n 1 -r reply
            echo ""
            if [[ $reply =~ ^[Yy]$ ]]; then
                git rm -r --cached "$folder_name" 2>/dev/null
                rm -rf "$local_repo_path"
                do_sparse_clone "$repo" "$local_repo_path" "$target_branch" "$remote_hash"
            else
                echo -e "${YELLOW}[-] Skipping update.${NC}\n"
            fi
        fi
    else
        echo -e "${GREEN}[!] Target folder untracked. Starting clean pull.${NC}"
        rm -rf "$local_repo_path"
        do_sparse_clone "$repo" "$local_repo_path" "$target_branch" "$remote_hash"
    fi
done