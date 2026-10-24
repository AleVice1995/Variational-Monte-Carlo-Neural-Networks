% --- identification
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{standard-doc}[2001/03/03 UoBnote class]

% --- using pkgs
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
%\RequirePackage[margin=0.5in]{geometry}
%\usepackage[margin=0.5in]{geometry}

% --- define options
\newif\if@color\@colorfalse
\newif\if@pdf\@pdffalse
\DeclareOption{color}{\@colortrue}
\DeclareOption{pdftex}{\@pdftrue}
\DeclareOption{twocolumn}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\if@pdf
\RequirePackage[pdftex]{graphicx}
\else
\RequirePackage{graphicx}
\fi

\LoadClass[a4paper]{article}

\topmargin -15mm
\headsep 10mm
\headheight 23mm

%%\footheight 0mm

\textheight230mm 
\textwidth150mm

%\textheight290mm 
%\textwidth190mm

\hoffset 0mm
\voffset 0mm
%%  modif centrage Coulon (papier A4, pas Letter) :
\oddsidemargin  -10mm
\evensidemargin -10mm

% These are not working
\leftmargin 15mm
\rightmargin 15mm


% --- load class


% color is switched on, use the color package
\if@color
\RequirePackage{color}
\fi%color


\renewcommand\maketitle{
\begin{titlepage}
	\centering % Center everything on the title page
	\scshape
\vspace*{1.5\baselineskip}

% ===================
%   Title Section
% ===================

% Double line above the title
\rule{13cm}{1.6pt}\\[-2pt]
\rule{13cm}{0.4pt}

\vspace{0.75\baselineskip}

% Title
{\fontsize{26pt}{31pt}\selectfont \@title\par}

\vspace{0.75\baselineskip}

% Double line below the title
\rule{13cm}{0.4pt}\\[-2pt]
\rule{13cm}{1.6pt}

\vspace{5\baselineskip}

% =================
%   Information
% =================

{\fontsize{23pt}{28pt}\selectfont \@author\par}

\vfill
\end{titlepage}
}


\pagestyle{fancy}
%\lhead{\setlength{\unitlength}{1mm}
%{\includegraphics[height=10mm]{bhamlogo.eps}}}

\chead{}
\rhead{}

\lfoot{} 
\cfoot{{\footnotesize \textit{page} \thepage\ of \pageref{LastPage}}} 
\rfoot{}


\message{note completed OK}


