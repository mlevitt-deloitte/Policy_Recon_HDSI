image=parkeraddison/policyrecon
devimage=parkeraddison/policyrecon-devcontainer
tag=latest

show_help() {
    echo "Usage: ./run.sh [TARGET] [-i/--image IMAGE] [-d/--dev-image DEVIMAGE] [-t/--tag TAG]\n"
    echo "Available Targets"
    echo "-----------------"
    grep ') #: ' run.sh | sed '1d' | sed 's,    \(.*\)) #: \(.*\),  \1#\2\n,' | column -ts '#'
    echo
}

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts hi:d:t:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    h | help )      show_help; exit 0 ;;
    i | image )     needs_arg; image="$OPTARG" ;;
    d | dev-image ) needs_arg; devimage="$OPTARG" ;;
    t | tag )       needs_arg; tag="$OPTARG" ;;
    ??* )           die "Illegal option --$OPT" ;;  # bad long option
    ? )             exit 2 ;;  # bad short option (error reported via getopts)
  esac
done

exe() { echo "\$ $@\n" ; "$@" ; }

case $1 in

    help) #: Show this help message. Default when no target provided.
        show_help
        ;;

    build) #: Build the base container image. Available options: image, tag
        exe docker build -t "${image}:${tag}" .
        ;;

    build-dev) #: Build the devcontainer image. Available options: dev-image, tag
        exe docker build -t "${devimage}:${tag}" -f .devcontainer/Dockerfile .
        ;;

    launch) #: Launch a base container image for easy manual pipeline runs. Available options: image, tag
    	exe docker run -it  \
            -v $(pwd):/workspace -w /workspace \
            -v $(pwd)/.docker-cache:/root \
            "${image}:${tag}" \
            bash
        ;;

    pipeline) #: Run pipeline from a base container image. Available options: image, tag
    	exe docker run -it  \
            -v $(pwd):/workspace -w /workspace \
            -v $(pwd)/.docker-cache:/root \
            "${image}:${tag}" \
            python pipeline.py
        ;;

    *)
        show_help
        ;;
esac
