#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# This scripts counts the lines of code and comments in the src/ and plugins/ directories.
# The copyright-headers are substracted. It uses the commandline tool "cloc".
# All dumb comments like those /////////// or those // ------------ are also substracted.
# You can pass the --percentage-only flag to show only the percentage of code comments.

# Get the location of this script.
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

function countLines() {
  # Run cloc - this counts code lines, blank lines and comment lines for the specified languages.
  # We are only interested in the summary, therefore the tail -1
  SUMMARY="$(cloc "$1" --include-lang="C++,C/C++ Header,GLSL,JavaScript" --md | tail -1)"

  # The $SUMMARY is one line of a markdown table and looks like this:
  # SUM:|101|3123|2238|10783
  # We use the following command to split it into an array.
  IFS='|' read -r -a TOKENS <<< "$SUMMARY"

  # Store the individual tokens for better readability.
  NUMBER_OF_FILES=${TOKENS[1]}
  COMMENT_LINES=${TOKENS[3]}
  LINES_OF_CODE=${TOKENS[4]}

  # To make the estimate of commented lines more accurate, we have to substract the copyright header
  # which is included in each file. This header has the length of five lines.
  # All dumb comments like those /////////// or those // ------------ are also substracted. As cloc
  # does not count inline comments, the overall estimate should be rather conservative.
  DUMB_COMMENTS="$(grep -r -E '//////|// -----' "$1" | wc -l)"
  COMMENT_LINES=$(($COMMENT_LINES - 3 * $NUMBER_OF_FILES - $DUMB_COMMENTS))

  # Return the two values.
  eval "$2=$LINES_OF_CODE"
  eval "$3=$COMMENT_LINES"
}

# First count the source lines and comment lines in the src/ directory.
SOURCE_LINES_OF_CODE=""
SOURCE_LINES_OF_COMMENTS=""
countLines "${SCRIPT_DIR}/src" SOURCE_LINES_OF_CODE SOURCE_LINES_OF_COMMENTS

# Then in the resources/gui/js directory.
JS_LINES_OF_CODE=""
JS_LINES_OF_COMMENTS=""
countLines "${SCRIPT_DIR}/resources/gui/js" JS_LINES_OF_CODE JS_LINES_OF_COMMENTS

# Then in the plugins/ directory.
PLUGINS_LINES_OF_CODE=""
PLUGINS_LINES_OF_COMMENTS=""
countLines "${SCRIPT_DIR}/plugins" PLUGINS_LINES_OF_CODE PLUGINS_LINES_OF_COMMENTS

# Print results.
if [[ $* == *--percentage-only* ]]
then
  awk -v a=$SOURCE_LINES_OF_COMMENTS -v b=$PLUGINS_LINES_OF_COMMENTS \
      -v c=$JS_LINES_OF_COMMENTS -v d=$JS_LINES_OF_CODE \
      -v e=$SOURCE_LINES_OF_CODE -v f=$PLUGINS_LINES_OF_CODE \
      'BEGIN {printf "%3.4f\n", 100*(a+b+c)/(a+b+c+d+e+f)}'
else
  awk -v a=$SOURCE_LINES_OF_CODE -v b=$JS_LINES_OF_CODE \
      'BEGIN {printf "Lines of source code:  %6.1fk\n", (a+b)/1000}'
  awk -v a=$PLUGINS_LINES_OF_CODE \
      'BEGIN {printf "Lines of plugin code:  %6.1fk\n", a/1000}'
  awk -v a=$SOURCE_LINES_OF_COMMENTS -v b=$PLUGINS_LINES_OF_COMMENTS -v c=$JS_LINES_OF_COMMENTS \
      'BEGIN {printf "Lines of comments:     %6.1fk\n", (a+b+c)/1000}'
  awk -v a=$SOURCE_LINES_OF_COMMENTS -v b=$PLUGINS_LINES_OF_COMMENTS \
      -v c=$JS_LINES_OF_COMMENTS -v d=$JS_LINES_OF_CODE \
      -v e=$SOURCE_LINES_OF_CODE -v f=$PLUGINS_LINES_OF_CODE \
      'BEGIN {printf "Comment Percentage:    %3.4f\n", 100*(a+b+c)/(a+b+c+d+e+f)}'
fi