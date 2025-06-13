#!/bin/bash
set -e
OS_VERSION=${OS_VERSION:-2.9.0}
if [ ! -d opensearch-$OS_VERSION ]; then
  curl -L -o os.tgz https://artifacts.opensearch.org/releases/bundle/opensearch/$OS_VERSION/opensearch-$OS_VERSION-linux-x64.tar.gz
  tar -xzf os.tgz
fi
opensearch-$OS_VERSION/opensearch-tar-install.sh &
# wait for OpenSearch to start
sleep 30
