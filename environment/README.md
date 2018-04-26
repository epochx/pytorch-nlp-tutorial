#  A collection of utils for comfortably developing on remote GPU-powered machines

## tmux

tmux is a terminal multiplexer that allows us to run multiple terminal sessions inside a single terminal window.  Besides being easily configurable,  the key feature for us is that tmux will let us detach from a terminal sesssion with programs running in it, while keeping the session running in the background. We can reattach to these sessions later from any other terminal.

If you have admin privileges, in ubuntu you can install `tmux` using `apt install tmux`, otherwise refer to  the [no_sudo_install_tmux script](https://gist.github.com/epochx/2abc52902ce99c2a3c8907ccad927581).

- The `.tmux.conf` file 

  - After installing,  copy this config  file into your remote home directory (usually `/home/username` in linux distros). If you already have a tmux config file, I'd recoomend you keep a backup .
  - Make sure to comment/uncomment the indicated lines depending on the version of tmux installed on your server. You can check the version by running `tmux -V`.
  - After this,  run `tmux source ~/.tmux.config` to load the settings.

- Using tmux:

  - Start tmux with the `tmux` command. This will create a new unnamed session.

  - To list existing sessions, call `tmux ls`.

  - To attach to an existing session  use `tmux a -t session_name`. For unnamed sessions you can use its numerical id instead.

  The config file is based on the shortcuts and other features available in `byobu`, which is a wrapper around `tmux` or `gnu-screen` .  Included is:

  - A status bar showing the open windows and curent date and  time.
  - Mouse support for scrolling, window selection, panel resizing and copy-pasting (hold shift for selecting the content you want to copy).

  Below I'm documenting some of the most useful shortcuts that are available. Please check the config file to learn more about this.

  | Shortcut   | Action                     |
  | ---------- | -------------------------- |
  | F2         | New window                 |
  | F3, F4     | Move to left, right window |
  | F6         | Detach from session        |
  | F8         | Rename session             |
  | alt + F2   | New vertical panel         |
  | shift + F2 | New horizontal panel       |

## SSH Tunneling

### Basic Friendly Configuration

- If you haven't created your own keys, run `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`. These credentials will be stored by default on the `/home/username/.ssh` folder.
- Open `/home/.ssh/config` with your favorite text editor and make sure you add aliases for all your favorite servers. You can use the `ProxyCommand` if you need to make double tunnels in order to access your server.
- Connect to your server  and add your public key, i.e. the contents of the file `/home/.ssh/id_rsa.pub` to the file `/home/.ssh/authorized_keys` in your server.

### Connecting to a remote Jupyter notebook

After connecting to my remote jupyter notebook instances every day for a while, I decided to create a shortcut function to help me do so. Besides having ssh aliases for your target servers, the only assumption for the system to work is that the jupyter instance is running on the 9999 port, which will be mapped to the 8080 local port. These port choices may seem odd, but they are intended to avoid port clashes with local jupyter instances that may be running, which would use port 8888 by default.

- Connect to your server, and launch jupyter with the command `jupyter notebook --port 9999`.
- use `start-remote-notebook SERVER_NAME` to connect to the server and point it to your local 8080 port. You can now open your browser, go to [http://localhost:8080/tree] and start working remotely.
- After you are finished working, you may use `kill-remote-notebook SERVER_NAME` to close the tunnel. Remember that the remote jupyter instance will stay open. If you keep it that way, all you have to do to reconnect is to use  the `start-remote-notebook` command again.
- You can point your tunnel to a different local port by adding the port parameter after the server name `start-remote-notebook SERVER_NAME LOCAL_PORT`.

### Mounting the remote filesystem

One last trick that I often use to make remote development easier is mounting the remote filesystem in my local machine. The effectiveness of this will ultimately depend on the quality of the network you are connected to, but in practice everything works well if you have a stable connection. Make sure you have installed `sshfs` (in macOS you may need to do a couple of extra steps) and that the folder `/mnt` exists.

- Use `mount-remote SERVER_NAME` to mount your remote home directory in the local path `/mnt/SERVER_NAME`.
- Use `umount-remote SERVER_NAME` to umount your remote mountpoint.

## Monitoring GPU usage and choosing between multiple GPUs

When developing or sharing a GPU machine, monitoring the GPU usage is one key aspect for development. To help with this process, I created a slightly modified version of the *gpustat* script, which offers features similar to *htop*, but focusing on GPU usage.
- paste the `gpustat.py` file in your home foder and call `gpustat` to launch a gpu monitor which inclides process id and user name identification.

On multiple GPU machines when using some DL libraries ---I'm looking at you,  Tensorflow---  it is a good idea to have finer control of the GPUs that will be available when running some code.  I've created a few bash functions to do that.
- You may use the command `get-visible-gpus` to obtain a list of the GPU devices that are visible to the .
- Use `set-visible-gpus `
- Use `get-total-gpus` to see the number of total GPUs available in the system, regardless of their visibility in the current session.
- Finally, use `reset-visible-gpus` to make all GPUs visible in the current session again.
