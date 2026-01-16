# Import tree to prevent circular imports

frxx:
 - core
 - io
 - proc
 - viz
 - misc

 - core:
 - - fdIQ
 - - fdMoments
 - - fdSpectra
 - - frxxData

 - - fdIQ, fdMoments, fdSpectra:
 - - - frxxData (only uses miscIO)

 - - frxxData:
 - - - io

 - io:
 - - miscIO
 - - decoders

 - - decoders:
 - - - decoder

 - - - decoder:
 - - - - readIQ

 - - - - readIQ
 - - - - - fdIQ (cycle here if core imported first)
