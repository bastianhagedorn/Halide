called=$_
[[ $called != $0 ]] || echo "jrk.bash should be sourced not run!"

function apps {
    cat apps.txt        | ruby -e "STDIN.read.split.each {|a| puts a+\"/${1}\"}"
    cat raceapps.txt    | ruby -e "STDIN.read.split.each {|a| puts a+\"/${1}\"}"
    cat tests.txt       | ruby -e "STDIN.read.split.each {|a| puts a+\"/${1}\"}"
}

appdir=$(dirname "${BASH_SOURCE[0]}")
export PATH=${PATH}:${appdir}

alias m=out-of-tree.sh

function gen_rand_id {
    LC_CTYPE=C tr -dc A-Za-z0-9 < /dev/urandom | fold -w ${1:-16} | head -n 1
}

function pushover {
    curl -s \
      --form-string "token=$PUSHOVER_APP_KEY" \
      --form-string "user=$PUSHOVER_USER_KEY" \
      --form-string '"'"$*"'"' \
      https://api.pushover.net/1/messages.json
}
