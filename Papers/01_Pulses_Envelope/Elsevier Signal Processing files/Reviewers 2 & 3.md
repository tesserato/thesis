# Reviewer #1: See attached file.

# Reviewer #2: 
The paper presents a method for estimating the temporal signal envelope by representing the signal in 2D space, defining the shape of the resulting object and calculating its discrete curvature.
There exist several methods to estimate the envelope of a signal, but most of them need an arbitrary decision concerning values of certain parameters. The authors claim, that a proposed plug-in replacement for many methods would be very useful. They recall several applications, where signal envelope estimation is indispensable. I agree, that such a method may be welcomed by researches.

The paper is rather compact, consists of the Abstract, four sections: Introduction, Methodology, Results, Discussion with Conclusions and References. The contents is illustrated by 11 Figures.

At first the paper seemed easy to read, but in-depth look has shown elements that need further clarification and a whole needs a structural arrangement.

First of all clear definition of terms used in the paper should be given at the beginning. It seems that the terms "frontier" and "envelope" are used interchangeably. In the abstract we read: "The approach draws inspiration from geometric concepts to isolate the frontiers and thus estimate the temporal envelope of an arbitrary signal". That would suggest that the meaning of these two terms is the same. But later the situation becomes more complicated, as in the next sentence of the abstract we read "We also define entities, such as a pulse and frontiers…" Is it the same "frontier" in both cases ? In Section 2. we read: "To that end, this work will emphasize the concept of frontier, as the points of the digital wave that mark the wave's upper and lower boundaries; from those points an envelope, conforming to the vague definitions of uniqueness and smoothness, can be trivially constructed, as will be shown." In Figure 3, indeed, we can see different lines representing the frontier and the envelope. But where this difference comes from? To make the situation worse, in the caption of the Figure 1 we read: "The dashed lines mark the frontiers between pseudo-cycles", whereas in the legend pseudo-cycles stand for dashed line.

Another question is, why the normalization according to formula (1) is necessary. The normalization parameter "s" is related to the number of pulses "m". If it is the number of pulses in the whole signal, then the envelope shape remain the same. But perhaps "m" may be arbitrarily chosen? Figure 8 would suggest it. It looks like the normalization was applied in separate parts of the signal. In such a case the shape of signal envelope is lost. In addition it is not clear why the representation in the frequency domain has been shown ? It seems that the subscriptions "Time domain" and "Frequency domain" have been swapped.

The highlight of the paper - the method of Discrete Curvature Estimation should be described more clearly and more systematically. Preferably the ALGORITHM should be provided - yet the Authors have practical applications in mind. Maybe there is a supplement demo or library available?

All the preparatory equations leading to the formula (2) should be written more clearly in separate lines and the parameters should be distinctly defined. This is core part of the paper, but very difficult to follow.

What worries me is the statement: "for an average wave of a few seconds at an ordinary rate of 44100 fps the average of the radius values is sufficient to assure the accurate identification of the frontier. However, a curve fitting method can be used in the case of longer or ill-behaved waves." What do you mean by "ill-behaved waves" ? It seems that the method will work for short signals, like individual notes played by the instruments. If so, this should be made obvious from the beginning of the paper.

My last comment concerns the placement of the fragment referring to the method of estimating signal's envelope using Hilbert transform - within the description of a new method in Section 2. Please, start the Section with this reference.

In general I would suggest the Authors to provide major changes in the text of the paper.



# Reviewer #3: 
This paper propose a method to estimate envelope of signals using intrinsic characteristics of the signals. I personally think that the topic of this paper is meaningful. However, the simulation example is relatively simple and not very convincing. The proposed method is relatively simple but time-consuming.
