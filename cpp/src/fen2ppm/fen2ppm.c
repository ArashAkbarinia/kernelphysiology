/*
fen2ppm.c Version 0.1.0 Chess Diagram from FEN file
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

/* Usage: fen2ppm <fen_diagram | pnmtopng >diagram.png */

/* You may use xboard to save a position as a FEN diagram. */
/* Click on file and save position. */
/* Type in filename.fen. */
/* The output of fen2ppm is a ppm file that is piped to */
/* a program that converts the ppm format to a compressed */
/* graphic image. */

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <zlib.h>
#include <stdlib.h>
#include <ctype.h>

#define FNT (49)
#define MAXROW (8)
#define MAXCOL (8)
#define FROWLEN (FNT * 3)
#define ROWLEN (FROWLEN*MAXROW)
#define FONTSZ (FROWLEN*FNT)
#define MTXSZ (FONTSZ*MAXROW*MAXCOL)
#define TBLSZ (2 * 2 * 7 * FNT * FROWLEN)
#define WHITE 0
#define BLACK 1
#define KING 0
#define QUEEN 1
#define BISHOP 2
#define KNIGHT 3
#define ROOK 4
#define PAWN 5
#define BLANK 6

int eofsw;

/* font table */
/* square color: grey, red */
/* piece  color: white, black */
/* piece */
/* 49 vertical pixels in font */
/* (49 horizontal pixels in font) * (3 bytes per pixel) */
unsigned char fnt[2][2][7][FNT][FROWLEN];

/* chess board */
/* 8 ranks */
/* 49 vertical pixels in font */
/* 8 columns */
/* (49 horizontal pixels in font) * (3 bytes per pixel) */
unsigned char mtx[MAXROW][FNT][MAXCOL][FROWLEN];

int getbyte()
{
  int rdlen;
  unsigned char buf[8];
  rdlen = read ( 0,buf,1 );
  if ( rdlen == 1 ) return ( buf[0] );
  else if ( !rdlen )
    {
      eofsw = 1;
      return ( EOF );
    } /* eof */
  else
    {
      perror ( "Read error" );
      exit ( 1 );
    } /* readerr */
} /* getbyte */

void putbuf ( int len )
{
  int wrtlen;
  wrtlen = write ( 1,mtx,len );
  if ( wrtlen == len ) return;
  else
    {
      fprintf ( stderr,"Write length %d\n", wrtlen );
      perror ( "Write error" );
      exit ( 1 );
    } /* write err */
} /* putbuf */

void isrtr ( unsigned char *dst, unsigned char *src )
{
  int i,j;
  unsigned char *p,*q;
  p = dst;
  q = src + FROWLEN*FNT;
  for ( i=0; i<FNT; i++ )
    {
      for ( j=0; j<FROWLEN; j++ )
        {
          *p++ = *q--;
        } /* for each pixel in font row */
      p += ( ROWLEN - FROWLEN );
    } /* for each row in font */
} /* isrt */

void isrt ( unsigned char *dst, unsigned char *src )
{
  int i,j;
  unsigned char *p,*q;
  p = dst;
  q = src;
  for ( i=0; i<FNT; i++ )
    {
      for ( j=0; j<FROWLEN; j++ )
        {
          *p++ = *q++;
        } /* for each pixel in font row */
      p += ( ROWLEN - FROWLEN );
    } /* for each row in font */
} /* isrt */

void loadtbl()
{
  int rdlen,stat;
  unsigned char *p;
  char fname[128];
  gzFile hndl;

  /* PTH is defined in fen2ppm.mak              */
  /* if local directory: PTH=\"\.\"             */
  sprintf ( fname,"%s/%s", PTH, "font49.gz" );
  hndl = gzopen ( fname,"rb" );
  if ( hndl == NULL )
    {
      fprintf ( stderr,"Error opening %s\n", fname );
      exit ( 1 );
    } /* open err */

  p = fnt[0][0][0][0];
  rdlen = gzread ( hndl,p,TBLSZ );
  if ( rdlen < 0 )
    {
      perror ( "Read error" );
      exit ( 1 );
    } /* read err */
  else if ( rdlen == 0 )
    {
      fprintf ( stderr,"Empty file: %s\n", fname );
      exit ( 1 );
    } /* read err */
  else if ( rdlen != TBLSZ )
    {
      fprintf ( stderr,"Wrong length %d: "
                "%s\n", rdlen, fname );
      exit ( 1 );
    } /* wrong len err */

  stat = gzclose ( hndl );
  if ( stat < 0 )
    {
      fprintf ( stderr,"Error closing %s\n", fname );
      exit ( 1 );
    } /* close err */
} /* loadtbl */

void ovflerr ( int row, int col, int ch )
{
  fprintf ( stderr,"Row overflow: row %d col %d char %c\n", row+1, col+1, ch );
  exit ( 1 );
} /* ovflerr */

void rowovfl ( int row, int col, int ch )
{
  fprintf ( stderr,"Column overflow: row %d col %d char %c\n", row+1, col+1, ch );
  exit ( 1 );
} /* rowovfl */

void piecerr ( int row, int col, int ch )
{
  fprintf ( stderr,"Row %d Col %d: Piece %c invalid\n", row+1, col+1, ch );
  exit ( 1 );
} /* piecerr */

int main ( int argc, char **argv )
{
  int eolsw,row,col;
  int i, j, k, ch, color, SideToMove;
  unsigned char *p,*q;
  char str[128];

  loadtbl();
  /* initialize chess board to blank squares */
  row = col = color = 0;
  for ( i=0; i<64; i++ )
    {
      q = fnt[color][WHITE][BLANK][0];
      p = mtx[row][0][col];
      for ( j=0; j<FNT; j++ )
        {
          for ( k=0; k<FROWLEN; k++ )
            {
              *p++ = *q++;
            } /* for each pixel in font row */
          p += ( ROWLEN - FROWLEN );
        } /* for each row in font */
      color ^= 1;
      col++;
      if ( col > 7 )
        {
          col = 0;
          row++;
          color ^= 1;
        } /* for each column */
    } /* for each blank square in row */

  // figure out who's to move
  FILE *file;
  file = fopen ( argv[1], "r" );
  ch = 0;
  while ( ( ch = getc ( file ) ) != ' ' )
    {
    }
  SideToMove = getc ( file );

  /* read the fen file */
  row = col = eofsw = 0;
  ch = color = 0;
  p = mtx[0][0][0];
  fseek ( file, 0, SEEK_SET );
  while ( ( ch = getc ( file ) ) != ' ' && ch != '\n' && ch != EOF )
    {
      if ( isdigit ( ch ) )
        {
          int blnk;
          blnk = ch - '0';
          if ( col + blnk > MAXCOL )
            {
              ovflerr ( row, col, ch );
            }
          col += blnk;
          if ( SideToMove == 'b' )
            {
              color = ( 7 - col ) % 2;
              if ( ( 7 - row ) % 2 )
                {
                  color ^= 1;
                }
            }
          else
            {
              color = col % 2;
              if ( row % 2 )
                {
                  color ^= 1;
                }
            }
        } /* numeric fen for blank squares */
      else if ( ch == '/' )
        {
          col = 0;
          row++;
          if ( row >= MAXROW )
            {
              rowovfl ( row, col, ch );
            }
          if ( SideToMove == 'b' )
            {
              color = ( 7 - row ) % 2;
              if ( ( 7 - col ) % 2 )
                {
                  color ^= 1;
                }
            }
          else
            {
              color = row % 2;
              if ( col % 2 )
                {
                  color ^= 1;
                }
            }
        } /* if end of row */
      else
        {
          int rotate = 0;
          switch ( ch )
            {
            case 'K':
              q = fnt[color][WHITE][KING][0];
              if ( SideToMove == 'b' )
                {
                  rotate = 1;
                }
              break;
            case 'Q':
              q = fnt[color][WHITE][QUEEN][0];
              if ( SideToMove == 'b' )
                {
                  rotate = 1;
                }
              break;
            case 'B':
              q = fnt[color][WHITE][BISHOP][0];
              if ( SideToMove == 'b' )
                {
                  rotate = 1;
                }
              break;
            case 'N':
              q = fnt[color][WHITE][KNIGHT][0];
              if ( SideToMove == 'b' )
                {
                  rotate = 1;
                }
              break;
            case 'R':
              q = fnt[color][WHITE][ROOK][0];
              if ( SideToMove == 'b' )
                {
                  rotate = 1;
                }
              break;
            case 'P':
              q = fnt[color][WHITE][PAWN][0];
              if ( SideToMove == 'b' )
                {
                  rotate = 1;
                }
              break;
            case 'k':
              q = fnt[color][BLACK][KING][0];
              if ( SideToMove == 'w' )
                {
                  rotate = 1;
                }
              break;
            case 'q':
              q = fnt[color][BLACK][QUEEN][0];
              if ( SideToMove == 'w' )
                {
                  rotate = 1;
                }
              break;
            case 'b':
              q = fnt[color][BLACK][BISHOP][0];
              if ( SideToMove == 'w' )
                {
                  rotate = 1;
                }
              break;
            case 'n':
              q = fnt[color][BLACK][KNIGHT][0];
              if ( SideToMove == 'w' )
                {
                  rotate = 1;
                }
              break;
            case 'r':
              q = fnt[color][BLACK][ROOK][0];
              if ( SideToMove == 'w' )
                {
                  rotate = 1;
                }
              break;
            case 'p':
              q = fnt[color][BLACK][PAWN][0];
              if ( SideToMove == 'w' )
                {
                  rotate = 1;
                }
              break;
            default:
              piecerr ( row, col, ch );
              break;
            } /* switch piece */

          int WhichRow = row;
          int WhichCol = col;
          if ( SideToMove == 'b' )
            {
              WhichRow = 7 - row;
              WhichCol = 7 - col;
            }
          p = mtx[WhichRow][0][WhichCol];
          if ( rotate == 1 )
            {
              isrtr ( p, q );
            }
          else
            {
              isrt ( p, q );
            }
          col++;
          if ( col > MAXCOL )
            {
              ovflerr ( row, col, ch );
            }
          color ^= 1;
        } /* not blank square */
    } /* read loop for fen */
  /* ppm header */
  sprintf ( str,"P6\n%d %d\n255\n", MAXROW*FNT, MAXCOL*FNT );
  printf ( "%s", str );
  fflush ( stdout );      /* synchronize output */
  /* ppm body */
  putbuf ( MTXSZ );
  return ( 0 );
} /* main */

