#!/bin/bash
vagrant ssh master -- -t "cd /rastercube; python setup.py bdist_egg; cd /terrapy; python setup.py bdist_egg"
