(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-environments-local "minted")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
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
    "sec:org595fbf9"
    "sec:orgedad357"
    "sec:orgd064d2e"
    "sec:org1956150"
    "sec:org2f7883c"
    "sec:org101599d"
    "sec:org25650c9"
    "sec:orge7bf8d1"
    "sec:org28e2809"
    "sec:org9647855"
    "sec:org1e03d6b"
    "sec:org7b48318"
    "sec:org877e0d1"
    "sec:org3279b62"
    "sec:org436b19a"
    "sec:org43dd9dc"
    "sec:org3e385a1")
   (LaTeX-add-bibliographies))
 :latex)

