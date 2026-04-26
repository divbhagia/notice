\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{div}
\LoadClass[12pt]{article}

% Multidocument processing 
%\RequirePackage{docmute}

% To avoid too many math letters error
\newcommand\hmmax{0}
\newcommand\bmmax{0}

% Misc Packages
\RequirePackage{comment}
\RequirePackage{enumitem}
\RequirePackage{etoolbox}
\RequirePackage[svgnames]{xcolor}

% Footnote symbols in title
\def\@fnsymbol#1{\ensuremath{\ifcase#1\or *\or \dagger\or \ddagger\or
   \mathsection\or \mathparagraph\or \|\or **\or \dagger\dagger
   \or \ddagger\ddagger \else\@ctrerr\fi}}

% Math
\RequirePackage{bm}
\RequirePackage{array}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsfonts}
\RequirePackage{amsthm}
\RequirePackage{mathtools}

% Font
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage{textcomp}
%\usepackage{mathpazo}

%\usepackage{mathdesign}
%\RequirePackage{microtype}
%\RequirePackage[adobe-utopia]{mathdesign} 
\RequirePackage[bitstream-charter]{mathdesign} 


% Figures/Graphics
\RequirePackage{morefloats}
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage[capposition=top]{floatrow}
\setlength{\belowcaptionskip}{0em}
\captionsetup{labelfont=sc, textfont=sc}
\captionsetup[floatfoot]{footfont={normal,footnotesize}, singlelinecheck=off, format=hang, width=\linewidth}

% Tables
\RequirePackage{multirow}
\RequirePackage{booktabs, siunitx}
\RequirePackage{tabularx}
\RequirePackage[para,flushleft]{threeparttable}
\renewcommand{\arraystretch}{1}	% spacing in tables
\AtBeginEnvironment{tablenotes}{\setstretch{1} \footnotesize}	% notes font
\newcolumntype{P}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{Y}{>{\centering\arraybackslash}X}


% Theorems & Assumptions
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{assump}{Assumption}
\newtheorem{prop}{Proposition}
\newtheorem{corr}{Corollary} 
\newtheorem*{assumpa*}{\assumptionnumber}
\providecommand{\assumptionnumber}{}
\newenvironment{assumpa}[2]
 {%
  \renewcommand{\assumptionnumber}{Assumption #1.a}%
  \begin{assumpa*}%
 }
 {%
  \end{assumpa*}
 }
\newtheorem*{assumpb*}{\assumptionnumber}
\providecommand{\assumptionnumber}{}
\newenvironment{assumpb}[2]
 {%
  \renewcommand{\assumptionnumber}{Assumption #1.b}%
  \begin{assumpb*}%
 }
 {%
  \end{assumpb*}
 }
% Expectation & Indicator Symbol
\newcommand{\E}{\index{Expected Value}\text{$\mathbb{E}$\kern.3pt}}
\newcommand{\I}{\index{Indicator Function}\text{$\mathbb{I}$\kern.3pt}}

% Colors
\definecolor{red}{RGB}{175 , 0 , 0}
\newcommand{\red}{\textcolor{red}}
\definecolor{blue}{RGB}{0 , 0 , 170}
\newcommand{\blue}{\textcolor{blue}}
% Spacing 
%\RequirePackage[nodisplayskipstretch]{setspace}
\RequirePackage{setspace}
\RequirePackage{indentfirst}
\setlength{\parindent}{1em}					% Paragraph Indentation
\setlength{\skip\footins}{1cm} 				% Footnote Space
\setlength{\parskip}{2mm} 						% Paragraph Skip
\setstretch{1.5} 								   % Line Space
\RequirePackage[margin=1.25in]{geometry} 	% Margins


% Sections & Subsections 
\RequirePackage{sectsty}
\RequirePackage{titlecaps}
\renewcommand{\thesection}{\Roman{section}} 
\renewcommand{\thesubsection}{\Roman{section}.\Alph{subsection}} 
\sectionfont{\large \centering \normalfont \sc}
\subsectionfont{\fontsize{13}{1} \normalfont \textit}

% Links
\RequirePackage{xr-hyper}
\RequirePackage[colorlinks, allcolors=NavyBlue]{hyperref}
\renewcommand\thefootnote{\textcolor{black}{\arabic{footnote}}}

% Citations
\RequirePackage{natbib}
\bibliographystyle{chicago}
\setcitestyle{authoryear,open={(},close={)}}

% References
\RequirePackage[nameinlink]{cleveref}
\Crefname{theorem}{Theorem}{Theorems}
\Crefname{assump}{Assumption}{Assumptions}
\Crefname{prop}{Proposition}{Propositions}
\Crefname{corr}{Corollary}{Corollaries}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Path & Directories                               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fix misplaced noalign error when compiling with pdflatex
\makeatletter
\ifdefined\pdfoutput
	\ExplSyntaxOn
	\cs_new:Npn \expandableinput #1  % providing \expandableinput
  	{ \use:c { @@input } { \file_full_name:n {#1} } }
  	\AddToHook{env/tabular/begin}  % add to tabular 
  	{ \cs_set_eq:NN \input \expandableinput }
	\AddToHook{env/tabularx/begin}  % add to tabularx
  	{ \cs_set_eq:NN \input \expandableinput }
	\ExplSyntaxOff
\else
\fi 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
