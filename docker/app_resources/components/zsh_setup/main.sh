#!/bin/bash

username=$1
cwd=$(dirname "$0")

sh -c "$(wget -O- \
  https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

git clone https://github.com/zsh-users/zsh-autosuggestions \
  ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-completions \
  ${ZSH_CUSTOM:-${ZSH:-~/.oh-my-zsh}/custom}/plugins/zsh-completions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git \
  ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-history-substring-search \
  ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search
bind "$terminfo[kcuu1]" history-substring-search-up
bind "$terminfo[kcud1]" history-substring-search-down


git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/powerlevel10k

fonts_dir="$(dirname "$0")/fonts"
install_dest="/usr/share/fonts/truetype/"
for font_file in "$fonts_dir"/*
do
  sudo install -m644 "$font_file" "$install_dest"
done

sudo chown "$username":"$username" "$cwd"/.p10k.zsh "$cwd"/.zshrc
sudo mv "$cwd"/.p10k.zsh "$cwd"/.zshrc /home/"$username"
echo exit | script -qec zsh /dev/null >/dev/null