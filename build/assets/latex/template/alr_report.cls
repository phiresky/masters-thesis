% Template for all kinds of theses at ALR (bachelor thesis, master thesis, ...)
% (c) Philipp Becker, 2020. 
% (Based on the 2019 H2T Template by Christian Dreher (based on the template from Stefan Ulbrich, 2012)

% We do need LaTeX2e
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{template/alr_report}
\LoadClass[a4paper,twoside,11pt,DIV=11,BCOR=15mm,bibliography=totoc,headings=twolinechapter,
parskip,numbers=endperiod]{scrreprt} % scrartcl

\special{pdf:minorversion 7}

% not using scrhack causes the following warning:
% % Class scrreprt Warning: \float@addtolists detected!
% % (scrreprt)              You should use the features of package `tocbasic'
% % (scrreprt)              instead of \float@addtolists.
% % (scrreprt)              Support for \float@addtolists may be removed from
% % (scrreprt)              `scrreprt' soon .
\usepackage{scrhack}

\ProcessOptions\relax

\RequirePackage{babel}
\usepackage[T1]{fontenc} % Encoding
\usepackage[utf8]{inputenc} % Encoding
\usepackage{graphicx}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
\usepackage{grffile}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{color}
\usepackage{subfigure}
\usepackage{tikz}
\usetikzlibrary{calc}
\usepackage{eso-pic}
\usepackage{booktabs}
\usepackage{fontspec}
% \usepackage[scaled=0.92]{helvet} % Different fonts
\setsansfont{Helvetica}
\usepackage{courier}
\setmonofont{Courier New}
\usepackage{mathptmx}
\usepackage{listings} % Source code listings, etc
\usepackage{lastpage}
\usepackage{microtype} % Better typesetting
\usepackage[automark,headsepline, draft=false]{scrlayer-scrpage}          
\usepackage{setspace}
\usepackage[inkscapearea=page, inkscapepath=figures/generated]{svg}

% hyperref userpackage moved down here because hyperref should be loaded as the
% last package: http://tex.stackexchange.com/questions/16268/warning-with-footnotes-namehfootnote-xx-has-been-referenced-but-does-not-exi
%\usepackage[raiselinks=true,
%bookmarks=true,
%bookmarksopenlevel=1,
%bookmarksopen=true,
%bookmarksnumbered=true,
%hyperindex=true,
%plainpages=false,
%pdfpagelabels=true,
%pdfborder={0 0 0.5},
%colorlinks=false,
%pagebackref=true,
%linkbordercolor={0 0.61 0.50},
%urlbordercolor={0 0.61 0.50},  % for same urlcolor
%citebordercolor={0 0.61 0.50},
%pdfcreator={LaTeX with ALR thesis template v0.0.0}]{hyperref}  %{0.57 0.74 0.57}
\usepackage[hidelinks]{hyperref}
\usepackage[noabbrev,nameinlink]{cleveref}  % Cleveref must be loaded after hyperref.

\definecolor{gray}{rgb}{0.5,0.5,0.5}

\lstloadlanguages{C++, bash,TeX,Python}
\lstset{
language={[LaTeX]TeX},
morekeywords={{\gqq},{includegraphics}}
}

\lstset{%
language=C++,
morekeywords={includegraphics,citet,citep},
showstringspaces=false,
basicstyle=\ttfamily\small,
keywordstyle=\color{blue},
commentstyle=\color{gray},
identifierstyle=\color{black},
stringstyle=\color{red},
captionpos=b,
frame=single,
frameround=tftf,
framesep=0.5cm,
xleftmargin=0.5cm,
xrightmargin=0.5cm,
rulecolor=\color{gray},float=htb,
breaklines=true,
%escapechar=\% %Enter latex commands in listing
}%

\newcommand{\termpaper}{\iflanguage{english}{Term Paper}{Seminararbeit}}
\newcommand{\mastersthesis}{\iflanguage{english}{Master's Thesis}{Masterarbeit}}
\newcommand{\bachelorsthesis}{\iflanguage{english}{Bachelor's Thesis}{Bachelorarbeit}}
\newcommand{\protocol}{\iflanguage{english}{Protocol}{Protokoll}}
\newcommand{\studienarbeit}{\iflanguage{english}{Student Research Project}{Studienarbeit}}
\newcommand{\diplomarbeit}{\iflanguage{english}{Diploma Thesis}{Diplomarbeit}}
\newcommand{\refereesingular}{\iflanguage{english}{Referee}{Gutachter}}
\newcommand{\refereeplural}{\iflanguage{english}{Referees}{Gutachter}}
\newcommand{\advisorsingular}{\iflanguage{english}{Advisor}{Betreuender Mitarbeiter}}
\newcommand{\advisorplural}{\iflanguage{english}{Advisors}{Betreuende Mitarbeiter}}

\newcommand{\mytitle} 
{\textcolor{red} {Please redefine the \texttt{mytitle} macro \\ \texttt{\Large\textbackslash renewcommand\{\textbackslash mytitle\}\{\dots\}}}}
\newcommand{\mythesis}
{\textcolor{red} {\small Please redefine the \texttt{mythesis} macro, \\ e.g., \texttt{\textbackslash renewcommand\{\textbackslash mythesis\}\{\textbackslash termpaper\}},\\ options: \texttt{\textbackslash mastersthesis, \textbackslash bachelorsthesis, \textbackslash studienarbeit, \textbackslash diplomarbeit}} }

%{\textcolor{red} {\large Please define the \texttt{mythesis} macro:  \texttt{ \textbackslash renewcommand\{\textbackslash mythesis\}\{\dots\}}
%\\ predefined macros are: \textbackslash termpaper, \textbackslash mastersthesis, \textbackslash bachelorsthesis, \textbackslash studienarbeit,
%\textbackslash dioplomarbeit 
%}}
\newcommand{\timestart}{\textcolor{red}{\small Please redefine the \texttt{timestart} macro: \texttt{ \textbackslash renewcommand\{\textbackslash timestart\}\{\dots\}}}}
\newcommand{\timeend}{\textcolor{red}{\small Please redefine the \texttt{timeend} macro:  \texttt{ \textbackslash renewcommand\{\textbackslash timeend\}\{\dots\}}}}
\newcommand{\myname}{\textcolor{red}{\large Please redefine the \texttt{myname} macro:  \texttt{ \textbackslash newcommand\{\textbackslash myname\}\{\dots\}}}}
\newcommand{\myshorttitle}{\textcolor{red}{\small Please redefine the \texttt{myshorttitle} macro (may be empty): \texttt{ \textbackslash newcommand\{\textbackslash myshorttitle\}\{\dots\}}}}

\newcommand{\gqq}[1]{\glqq{}#1\grqq{}}
\newcommand{\gq}[1]{\glq{}#1\grq{}}

\newcommand{\myinstitute}{
	\iflanguage{english}{Institute for Anthropomatics and Robotics}{Institut für Anthropomatik und Robotik} (IAR)\\
	\iflanguage{english}{Autonomous Learning Robots}{Autonome Lernende Roboter} (ALR)}

\newcommand{\myabstract}{}
\newcommand{\mysummary}{}

\newcommand{\abstracttext}[1]{\renewcommand{\myabstract}{#1}}
\newcommand{\summarytext}[1]{\renewcommand{\mysummary}{#1}}

% Page settings
\ihead[]{}
\ifoot[]{}
\ofoot[]{}

\lehead[]{\textnormal{\bfseries\sffamily \thepage}}
\rehead[]{\textnormal{\bfseries\sffamily \leftmark}}
\lohead[]{\textnormal{\bfseries\sffamily \rightmark}}
\rohead[]{\textnormal{\bfseries\sffamily \thepage}}

%\ohead[]{\textnormal{\sffamily \pagemark}}

\cfoot[\textnormal{\bfseries\sffamily \thepage}]{}


\addto\captionsenglish{%
    \renewcommand{\contentsname}%
    {Table of Contents}%
}

% Left/Right mark cannot be influenced directly:
\newcommand{\sectionname}{\iflanguage{english}{Section}{Abschnitt}}
\renewcommand{\chaptermark}[1]{\markboth{\textsf{\chaptername{} \thechapter.~#1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\textsf{\sectionname{} \thesection.~#1}}{}}

\pagestyle{scrheadings} % Commit changes
%\renewcommand{\chapterpagestyle}{scrheadings}

\definecolor{hdrs}{HTML}{555555}

\renewcommand*{\chapterformat}{%
    \mbox{\LARGE\color{hdrs}{\chapappifchapterprefix{\nobreakspace}\thechapter}%
        \autodot\IfUsePrefixLine{}{\enskip}}}
\renewcommand*{\sectionformat}{\color{hdrs}{\thesection}\autodot\enskip}
\renewcommand*{\subsectionformat}{\color{hdrs}{\thesubsection}\autodot\enskip}

\renewcommand*{\chapterheadstartvskip}{\vspace*{5cm}}
\renewcommand*{\chapterheadendvskip}{\vspace{1.5cm}}

% Define sections (front-, main- and backmatter)
\newcommand{\frontmatter}{\pagenumbering{roman}}
\newcommand{\mainmatter}{\pagenumbering{arabic}\setcounter{page}{1}}
\newcommand{\backmatter}{%
    \renewcommand*{\chaptermark}[1]{\markboth{\textsf{##1}}{\textsf{##1}}}
    %\renewcommand*{\sectionmark}[1]{\markright{\textsf{\sectionname{} \thesection.~##1}}{}}
}

% Penalize widows and clubs.
\widowpenalty10000
\clubpenalty10000


%%%%%%%%%%%%% TITLEPAGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\maketitle}{
\begin{titlepage}
\thispagestyle{empty}
\definecolor{cffffff}{RGB}{255,255,255}
\definecolor{c231f20}{RGB}{35,31,32}
\definecolor{c00a990}{RGB}{0,169,144}


\begin{tikzpicture}[remember picture,yscale=-1,x=1cm,y=1cm,overlay,shift=(current page.north west)]
\coordinate (BCOR) at (1.5cm,0.cm);
\coordinate (upperleft) at ($(BCOR)+(1.5cm,1.5cm)$);
\coordinate (upperright) at ($(\paperwidth-1.5cm,1.5cm)$);
\coordinate (lowerleft) at ($(BCOR)+(1.5cm,\paperheight-2.0cm)$);
\coordinate (lowerright) at ($(\paperwidth-1.5cm,\paperheight-2.0cm)$);
\coordinate (logoKIT) at ($(upperleft) + (0.75cm,0.75cm)$);
\coordinate (logotext) at ($(logoKIT) + (0,1.8cm)$);
\coordinate (logoH2T) at ($(upperleft) + (0.75cm,0.75cm)$);
\coordinate (footnote) at ($(BCOR) + (1.25cm, \paperheight - 1.5cm)$);
\coordinate (footnote2) at ($(BCOR) + (1.48cm + \textwidth, \paperheight - 1.5cm)$);
  
\node [outer sep=0pt, anchor=west] at (footnote) {
\textbf{\textsf{
  \iflanguage{english}
  {\scriptsize KIT --- The Research University in the Helmholtz Association}
  {\scriptsize KIT --- Die Forschungsuniversität in der Helmholtz-Gemeinschaft}
}}
};

\node [outer sep=0pt, anchor=west] at (footnote2) {
\textsf{\large{\textbf{www.kit.edu}}}
};

\draw[thick,black,rounded corners=0.4cm]
(upperleft) -- (upperright) -- (lowerright); 
\draw[thick,black,rounded corners=0.4cm]
(upperleft) -- (lowerleft) -- (lowerright); 

% Text in the middle
\node[text width = \paperwidth-5cm] at ($(current page.center)+0.5*(BCOR)$) {
%
\sffamily
\bfseries
~\\[3cm]
\begin{center}
\Huge
\begin{spacing}{1.3}
\mytitle\\
\Large
\myshorttitle%
\end{spacing}
~\\[0cm]
\Large 
\mythesis%
\\
\iflanguage{english}{of}{von} \\[0.5cm]
\Huge 
\myname%
\\[1cm]
\large
\iflanguage{english}{KIT Department of Informatics}{KIT-Fakultät für Informatik}\\
\myinstitute\\[4cm]

\ifdefined \refereetwo \let\refereetext\refereeplural \else \let\refereetext\refereesingular \fi 
\ifdefined \advisortwo \let\advisortext\advisorplural \else \let\advisortext\advisorsingular \fi 

  
  \begin{tabular}[ht]{rl}
  \refereetext{}:  & \referee{}\\
  & \ifdefined \refereetwo \refereetwo \fi \\
  & \ifdefined \refereethree \refereethree \fi \\

  \addlinespace[0.6cm]

   \advisortext{}: &  \advisor{}\\
   & \ifdefined \advisortwo \advisortwo \fi \\
   & \ifdefined \advisorthree \advisorthree \fi \\
\end{tabular}\\[2cm]
\large 
\vfill
\iflanguage{english}{Duration}{Bearbeitungszeit}:
\timestart{} \hspace*{0.05cm} --- \hspace*{0.05cm} \timeend{}
\end{center}
  };
   
%The logo exported from Inksape    
%\begin{scope}[shift=(logoKIT),scale=0.12]
%  \path[fill=c231f20,even odd rule] (22.0818,12.6820) -- (18.2146,12.6820) --
%    (13.6183,7.7472) -- (13.6183,4.9217) -- (18.0714,-0.0000) -- %(22.0557,-0.0000)
%    -- (16.9256,6.3800) -- (22.0818,12.6820) -- cycle(22.6677,0.0129) --
%    (26.1442,0.0129) -- (26.1442,12.6820) -- (22.6677,12.6820) -- %(22.6677,0.0129)
%    -- cycle(35.3499,0.0129) -- (26.9776,0.0129) -- (26.9776,2.8644) --
%    (29.4255,2.8644) -- (29.4255,12.6952) -- (32.9020,12.6952) -- %(32.9020,2.8644)
%    -- (35.3499,2.8644) -- (35.3499,0.0129);
%  \path[fill=c231f20,nonzero rule] (12.7850,12.6820) -- (12.7850,12.6820);
%  \path[fill=c231f20,nonzero rule] (12.7850,12.6820) -- (12.7850,12.6820);
%  \path[fill=c00a990,even odd rule] (2.8763,4.6484) .. controls (2.2122,5.4557)
%    and (1.6393,6.3411) .. (1.1966,7.2914) -- (12.7850,12.6820) --
%    (2.8763,4.6484);
%  \path[fill=c00a990,even odd rule] (12.7850,12.6820) -- (7.3814,1.1847) ..
%    controls (6.4179,1.6276) and (5.5325,2.1743) .. (4.7252,2.8255) --
%    (12.7850,12.6820);
%  \path[fill=c231f20,even odd rule] (12.7850,12.6820) -- (12.7980,0.0129) ..
%    controls (12.7590,0.0129) and (12.7329,0.0129) .. (12.6939,0.0129) .. %controls
%    (11.6652,0.0129) and (10.6626,0.1433) .. (9.6991,0.3775) -- %(12.7850,12.6820);
%  \path[fill=c231f20,even odd rule] (12.7850,12.6820) -- (12.7850,12.6820);
%  \path[fill=c231f20,even odd rule] (12.7850,12.6820) -- (12.7850,12.6820);
%  \path[fill=c00a990,even odd rule] (0.3633,9.5833) .. controls %(0.1028,10.5728)
%    and (-0.0274,11.6143) .. (-0.0404,12.6820) -- (12.7850,12.6820) --
%    (0.3633,9.5833);
%\end{scope}

\node[inner sep=0pt] (KIT) at ($(5.8cm,3.18cm)$) {
\iflanguage{english}{
    \includegraphics[height=1.87cm]{template/logo_kit_en}
}{
    \includegraphics[height=1.87cm]{template/logo_kit_de}
}
};

%TODO: Add Logo here once available!

%\node[inner sep=0pt] (H2T) at ($(\paperwidth-4cm,3.05cm)$) {
%\iflanguage{english}{
   % \includegraphics[height=2.1cm]{ENGLISH_LOGO}
%}{
   % \includegraphics[height=2.1cm]{GERMAN_LOGO}
%};
    
\end{tikzpicture}
\end{titlepage}

\cleardoublepage
}


\newcommand{\makedeclaration}
{
% Backup currently used language and set ngerman for this page.
\let\oldlanguage\languagename
\selectlanguage{ngerman}

\thispagestyle{plain}
\newpage
%\topskip0pt
\vspace*{\fill}
\noindent
\textbf{Plagiarism Declaration (German)}
\vspace*{1em}

\noindent
Ich versichere hiermit, dass ich die Arbeit selbstständig verfasst habe, keine anderen als die angegebenen Quellen und Hilfsmittel benutzt habe, die wörtlich oder inhaltlich übernommenen
Stellen als solche kenntlich gemacht habe und die Satzung des Karlsruher Instituts für Technologie zur Sicherung guter wissenschaftlicher Praxis beachtet habe.
\vspace*{2em}

\noindent
Karlsruhe, den \timeend{}
\begin{flushright}
    $\overline{\mbox{\myname{}}}$
\end{flushright}

% Restore previously used language.
\expandafter\selectlanguage\expandafter{\oldlanguage}

\vspace*{\fill}
\cleardoublepage
}

% Load alr specific definitions.
\input{template/alr_def}
