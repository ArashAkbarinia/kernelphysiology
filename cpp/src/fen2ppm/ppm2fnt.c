/*
ppm2fnt.c Version 0.1.0 Convert PPM format to FNT format.
Copyright (C) 2003  dondalah@ripco.com (Dondalah)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to:

	Free Software Foundation, Inc.
	59 Temple Place - Suite 330
	Boston, MA  02111-1307, USA.
*/

/* Usage: ppm2fnt <type.ppm >type.fnt */

/* The output of ppm2fnt is a .fnt file that is used by */
/* fen2ppm to create a go diagram */

/* This program strips the header from the .ppm file */
/* What remains is the font matrix */

#include <stdio.h>

#define BUFLEN (1024*1024)

int eofsw;

int getbyte()
   {
   int rdlen;
   unsigned char buf[8];
   rdlen = read(0,buf,1);
   if (rdlen == 1) return(buf[0]);
   else if (!rdlen)
      {
      eofsw = 1;
      return(EOF);
      } /* eof */
   else
      {
      perror("Read error");
      exit(1);
      } /* readerr */
   } /* getbyte */

void putbuf(len,buf)
int len;
unsigned char *buf;
   {
   int wrtlen;
   wrtlen = write(1,buf,len);
   if (wrtlen == len) return;
   else
      {
      perror("Write error");
      exit(1);
      } /* write err */
   } /* putbuf */

int main()
   {
   int ch,i,len;
   unsigned char *p;
   unsigned char buf[BUFLEN];
   for (i=0;i<13;i++) getbyte();
   p = buf;
   while (!eofsw)
      {
      ch = getbyte();
      if (ch != EOF) *p++ = ch;
      } /* read loop */
   len = p - buf;
   putbuf(len,buf);
   return(0);
   } /* main */
