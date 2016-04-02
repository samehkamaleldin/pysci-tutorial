#!/bin/sh

# ==============================================================================
# = project       :- pysci-tutorial
# = module        :- bootstrap shell script
# = author        :- sameh kamal
# = description   :- provisioning shell script of ubuntu machine
# = preconditions :- runs on ubuntu 14.4.x
# ==============================================================================

# update ubuntu repositories
sudo apt-get update

sudo apt-get install python3
sudo apt-get install pip3
