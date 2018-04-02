#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <ctype.h>

#define MAXROW (8)
#define MAXCOL (8)

void ovflerr ( int row, int col, int ch )
{
  fprintf ( stderr,"Row overflow: row %d col %d char %c\n", row, col, ch );
  exit ( 1 );
}

void rowovfl ( int row, int col, int ch )
{
  fprintf ( stderr,"Column overflow: row %d col %d char %c\n", row, col, ch );
  exit ( 1 );
}

void piecerr ( int row, int col, int ch )
{
  fprintf ( stderr,"Row %d Col %d: Piece %c invalid\n", row, col, ch );
  exit ( 1 );
}

void blanks ( unsigned char board[8][8], int row, int col, int ch, int SideToMove )
{
  int WhichRow = row;
  int WhichCol = 0;
  char bow;
  if ( SideToMove == 'b' )
    {
      WhichRow = 7 - row;
    }
  for ( int i = 0; i < ch; i++ )
    {
      if ( SideToMove == 'b' )
        {
          WhichCol = 7 - ( col + i );
        }
      else
        {
          WhichCol = col + i;
        }

      if ( WhichRow % 2 == 0 )
        {
          if ( WhichCol % 2 == 0 )
            {
              bow = '.';
            }
          else
            {
              bow = 'x';
            }
        }
      else
        {
          if ( WhichCol % 2 == 0 )
            {
              bow = 'x';
            }
          else
            {
              bow = '.';
            }
        }
      board[WhichRow][WhichCol] = bow;
    }
}

int main ( int argc, char **argv )
{
  unsigned char board[8][8];
  int row, col;

  // figure out who's to move
  FILE *file;
  file = fopen ( argv[1], "r" );
  int ch = 0;
  while ( ( ch = getc ( file ) ) != ' ' )
    {
    }
  int SideToMove = getc ( file );

  // reading the fen file
  fseek ( file, 0, SEEK_SET );
  row = col = 0;
  while ( ( ch = getc ( file ) ) != ' ' && ch != '\n' && ch != EOF )
    {
      if ( isdigit ( ch ) )
        {
          ch = ch - '0';
          if ( col + ch > MAXCOL )
            {
              ovflerr ( row, col, ch );
            }
          blanks ( board, row, col, ch, SideToMove );
          col += ch;
        } /* numeric fen for blank squares */
      else if ( ch == '/' )
        {
          col = 0;
          row++;
          if ( row >= MAXROW )
            {
              rowovfl ( row, col, ch );
            }
        } /* if end of row */
      else
        {
          switch ( ch )
            {
            case 'K':
              break;
            case 'Q':
              break;
            case 'B':
              break;
            case 'N':
              break;
            case 'R':
              break;
            case 'P':
              break;
            case 'k':
              break;
            case 'q':
              break;
            case 'b':
              break;
            case 'n':
              break;
            case 'r':
              break;
            case 'p':
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
          board[WhichRow][WhichCol] = ch;
          col++;
          if ( col > MAXCOL )
            {
              ovflerr ( row, col, ch );
            }
        } /* not blank square */
    } /* read loop for fen */

  for ( unsigned int i = 0; i < MAXROW; i++ )
    {
      for ( unsigned int j = 0; j < MAXCOL; j++ )
        {
          printf ( "%d ", (int) board[i][j] );
        }
      printf ( "\n" );
    }

  return 0;
} /* main */
