#!/usr/bin/env bash

# copy this into your ~/.bashrc

start-remote-notebook() {
    # do things with parameters like $1 such as
    # assumes remote port is 9999 and local is 8080
    SERVER=$1
    PORT=${2-8080}
    ssh -fNL $PORT:localhost:9999 $SERVER
}

kill-remote-notebook(){
    SERVER=$1
    ps -ef | grep "9999 $SERVER" | grep -v grep | awk '{print $2}' | xargs kill -9
}


# Tensorboard tools

start-remote-tensorboard() {
    # do things with parameters like $1 such as
    # assumes remote port is 9999 and local is 8080
    SERVER=$1
    PORT=${2-7007}
    ssh -fNL $PORT:localhost:6006 $SERVER
}

kill-remote-tensorboard(){
    SERVER=$1
    ps -ef | grep "7007 $SERVER" | grep -v grep | awk '{print $2}' | xargs kill -9
}


# Remote developing tools

mount-remote() {
    SERVER=$1
    if [ ! -d /mnt/"$SERVER" ]; then
      sudo mkdir /mnt/"$SERVER"
      sudo chown "$USER" /mnt/"$SERVER"
    fi

    sshfs "$SERVER": /mnt/"$SERVER"

}

umount-remote() {
    SERVER=$1
    # kill bash tunnel
    #ps -ef | grep "22 $SERVER" | grep -v grep | awk '{print $2}' | xargs kill -9
    # umount
    sudo umount /mnt/"$SERVER"
}


# autocompletion support for ubuntu

_myssh() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts=$(grep '^Host' ~/.ssh/config | grep -v '[?*]' | cut -d ' ' -f 2-)

    COMPREPLY=( $(compgen -W "$opts" -- ${cur}) )
    return 0
}

_mysshsync() {
    local cur prev opts
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    diropts=$( ls "/home/edison/PycharmProjects/")
    diropts+=" data"

    if [ $prev == "sync-project" ]; then
        COMPREPLY=( $(compgen -W "$diropts" -- ${cur}) )
        return 0
    else
        _myssh
        return 0
    fi
}

}

complete -F _myssh start-remote-notebook
complete -F _myssh kill-remote-notebook
complete -F _myssh start-remote-tensorboard
complete -F _myssh kill-remote-tensorboard
complete -F _myssh mount-remote
complete -F _myssh umount-remote
complete -F _mysshsync sync-project
