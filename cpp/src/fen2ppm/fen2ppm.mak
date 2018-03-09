#!/usr/bin/make

# fen2ppm.mak Version 0.1.0 Compile fen2ppm.c
# Copyright (C) 2003  dondalah@ripco.com (Dondalah)
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to:
# 
# 	Free Software Foundation, Inc.
# 	59 Temple Place - Suite 330
# 	Boston, MA  02111-1307, USA.

# directory path for /foo/bar/font49.gz
# PTH=\"/foo/bar\"
# if local directory: PTH=\"\.\"

#PTH=\"\.\"
PTH=\"/home/arash/Software/binaries/fen2ppm-0.1.0\"

CC=gcc

CFLAGS=-DPTH=$(PTH)

LDFLAGS=-L/usr/lib -lz

fen2ppm:	fen2ppm.c fen2ppm.mak
	$(CC) fen2ppm.c \
	$(CFLAGS) \
	-o fen2ppm $(LDFLAGS)

clean:
	rm -f fen2ppm
