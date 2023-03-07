# PolicyRecon NLP Hackathon - Halıcıoğlu's Prodigies

<img src="img/abstract-logo.png" align="right">

This repo explores the ability to detect contradictions in government policy documents using natural language processing (NLP). The problem statement as posed in the PolicyRecon hackathon which inspired this repo is as follows:
> Given a collection of policy documents, identify contradictions (statement in one policy are contradicted by another policy in the same group or org) within that collection



**Table of Contents**
- [Setup](#setup)
  - [Data](#data)
  - [Environment](#environment)
    - [To Install Additional Pip Packages](#to-install-additional-pip-packages)

## Setup

### Data
Please download the datasets from the "PolicyRecon Analytics Challenge 2023" Team under *General > Files > 02. Data Sets* and save it to `data/02. Data Sets/` so that we all have the data in the same location.

### Environment
We can use Dev Containers to make setting up our development environment easy. Under `.devcontainer/` we define our Dockerfile and requirements.txt. To launch a container with our environment set up do the following steps in VSCode:
1. Install Docker on your system https://docs.docker.com/get-docker/
2. Install VSCode on your system https://code.visualstudio.com/download
3. Launch VSCode
   1. Install the "Dev Containers" (ms-vscode-remote.remote-containers) extension from Microsoft
   2. Click the pop-up window that says "Reopen in container" *[or]* use the command palette (<kbd>Ctrl/Cmd+Shift+P</kbd>) and search for then run the command "**Dev Containers: Reopen in Container**"

The window will then reload into a dev container, build the image and install the Python packages if this is your first time running, then allow you to run notebooks and terminal commands from within the Linux-based container, mounted to this repository.

#### To Install Additional Pip Packages
1. Launch the dev container
2. Install the package via integrated terminal
3. Run `pip freeze | grep <packagename>` to find the version installed and add it to our requirements.txt file
4. Add any additional commands you had to run in the terminal (such as additional non-pip software dependencies that had to be installed) to our Dockerfile in a new RUN command
5. Run the VSCode command "**Dev Containers: Rebuild and Reopen in Container**" to make sure it works
6. Commit and push your changes to our git repo
