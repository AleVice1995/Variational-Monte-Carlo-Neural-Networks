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
	\scshape % Use small caps for all text on the title page
	\vspace*{1.5\baselineskip} % White space at the top of the page
% ===================
%	Title Section 	
% ===================

	\rule{13cm}{1.6pt}\vspace*{-\baselineskip}\vspace*{2pt} % Thick horizontal rule
	\rule{13cm}{0.4pt} % Thin horizontal rule
	
		\vspace{0.75\baselineskip} % Whitespace above the title
% ========== Title ===============	
	{	\Huge Extensive Guide\\ 
			\vspace{4mm}
		to \LaTeX \\	}
% ======================================
		\vspace{0.75\baselineskip} % Whitespace below the title
	\rule{13cm}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt} % Thin horizontal rule
	\rule{13cm}{1.6pt} % Thick horizontal rule
	
		\vspace{1.75\baselineskip} % Whitespace after the title block
% =================
%	Information	
% =================
	{\large Produced by: Armin '19 \\
		\vspace*{1.2\baselineskip}
	armindubert2019@gmail.com} \\
	\vfill
If you come across any problems, see section \ref{sec:resources} for possible\\ \vspace{1mm}
solutions or contact me at \url{armindubert19@gmail.com}\\ \vspace{1mm}
Happy \LaTeX-ing!
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


