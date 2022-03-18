#!/bin/bash
qsub controlscript.pbs 70 120 1

qsub controlscript.pbs 70 120 2

qsub controlscript.pbs 140 240 2

qsub controlscript.pbs 140 240 4

qsub controlscript.pbs 140 240 6
