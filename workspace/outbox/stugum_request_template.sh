#!/usr/bin/env bash
set -euo pipefail

# Safe template generated from the structure of dp1.txt.
# Fill these values locally only if you are authorized to use them
# and the website permits this kind of request.

URL="https://stugum.emis.am/home/check"

RECAPTCHA_TOKEN="PUT_RECAPTCHA_TOKEN_HERE"
PARENT_SSN="PUT_PARENT_SSN_HERE"
PARENT_FIRST_NAME="PUT_PARENT_FIRST_NAME_HERE"
PARENT_LAST_NAME="PUT_PARENT_LAST_NAME_HERE"
CHILD_SSN="PUT_CHILD_SSN_HERE"
SCHOOL_STD_SSN=""
CSRF_TOKEN=""

curl "$URL" \
  -X POST \
  -H "accept: application/json, text/javascript, */*; q=0.01" \
  -H "accept-language: ru-RU,ru;q=0.9,en-GB;q=0.8,en;q=0.7,en-US;q=0.6,hy;q=0.5" \
  -H "x-requested-with: XMLHttpRequest" \
  -H "referer: https://stugum.emis.am/" \
  -F "ci_csrf_token=${CSRF_TOKEN}" \
  -F "ssn_checker_g-recaptcha-response=${RECAPTCHA_TOKEN}" \
  -F "p_ssn=${PARENT_SSN}" \
  -F "parent_first_name=${PARENT_FIRST_NAME}" \
  -F "parent_last_name=${PARENT_LAST_NAME}" \
  -F "ch_ssn=${CHILD_SSN}" \
  -F "school_std_ssn=${SCHOOL_STD_SSN}"
