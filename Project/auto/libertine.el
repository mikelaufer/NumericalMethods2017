(TeX-add-style-hook
 "libertine"
 (lambda ()
   (TeX-run-style-hooks
    "LinLibertine_R"
    "LinBiolinum_R"
    "LinBiolinum_K"
    "LinLibertine_I"
    "LKey"
    "ifxetex"
    "ifluatex"
    "xkeyval"
    "textcomp"
    "fontspec"
    "mweights"
    "fontenc"
    "fontaxes")
   (TeX-add-symbols
    '("libertineInitialGlyph" 1)
    '("biolinumKeyGlyph" 1)
    '("biolinumGlyph" 1)
    '("libertineGlyph" 1)
    '("DeclareTextGlyphY" 3)
    "LinuxLibertineT"
    "LinuxLibertineDisplayT"
    "LinuxBiolinumT"
    "LinuxLibertineMonoT"
    "LinuxLibertineInitialsT"
    "useosf"
    "libertineSB"
    "libertineOsF"
    "libertineLF"
    "libertineDisplay"
    "biolinum"
    "biolinumOsF"
    "biolinumLF"
    "libmono"
    "libertineInitial"
    "textsu"
    "oldstylenums"
    "liningnums"
    "oldstylenumsf"
    "liningnumsf"
    "tabularnums"
    "proportionalnums"
    "tabularnumsf"
    "proportionalnumsf"
    "rmdefault"
    "sfdefault"
    "ttdefault"
    "sufigures"
    "textsuperior"))
 :latex)

