(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-environments-local "minted")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "{preamble"
    "article"
    "art11"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "minted"
    "libertine"
    "lmodern"
    "natbib")
   (LaTeX-add-labels
    "sec:org575075d"
    "sec:org380d714"
    "sec:org8e6c72c"
    "sec:org5d5c538"
    "sec:orga301617"
    "sec:orgfd70d51"
    "sec:org1d2240d"
    "sec:org1010037"
    "sec:org6c35d19"
    "sec:org889dec9"
    "sec:org01e5289"
    "sec:org1ef16e2"
    "sec:org1051a86"
    "sec:orge51f918"
    "sec:orgd8f4e75"
    "sec:org7b80417")
   (LaTeX-add-bibliographies))
 :latex)

