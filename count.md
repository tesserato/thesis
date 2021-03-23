% Tell the reader the problem you are tackling in this project.
The area of artificial intelligence and, more specifically, neural networks, is interdisciplinar since its conception, being born from biological inspirations. Implementation wise, connections with fields like programming, computing and mathematics imposed themselves since the begining.

Attention, however, seems to revolver around the use of advancements in related disciplines to the betterment of neural networks theory in general, be it in the form of convolutional networks, better hyperparameter search, parallel implementations, etc.
Generally, neural networks are applyied to some problem, with little concern about transforming the problem in order to make it optimally suited to work in a neural networks approach, outside of necessary adaptations, such as normalization.

This lack is in part justified by the recent progress exhibted in the field, where impressive results are abundant; a word of caution is needed here, however, as this is a field that experimented its fair share of hype cycles, followed by winters.

% State clearly how you aim to deal with this problem. 

In this context, the primary objective of this work is to illustrate how such integration can be done, by developing a new sound representation designed specifically to be used by neural networks in sound synthesis tasks.

Since the beginning of computing, even when mainframes were the only available medium, restricted to organizations with considerable budgets, people found ways and motivation to overcome the technical limitations and develop artistic applications, like video games, in their (and the machine's) spare time. 


On a parallel note, also relevant to this thesis, those first machines, like the \gls{ENIAC} in 1946, brought with then the promise of the artificial intelligence as their ultimate goal \parencite{2010DonovanReplay}.

Music, on the other hand, is another staple of humanity's creativity. % TODO first musical instruments
 Is only natural that, as soon as this technology became reasonably stable, musical applications started to emerge.

Ambitious as it may seem, it is the goal of this monograph to motivate a change of approach in domains that, despite their modest intersections, share more attributes than is apparent at a superficial glance: Both machine learning and digital musical acoustics are areas that, despite their current relevance, are fresh out of their respective infancies, and a scrutiny of both promptly reveals the difficulties that arise from this state of affairs. A lack of their own terminology, with frequent borrows from other, more established areas, can be readily cited, as the pronounced prominence of few individual contributions.

Those shortcomings have, however, a bright side to them, in that they still expose those areas to reinterpretation, rendering them somewhat open to the influence of outside ideas.

Digital musical acoustics borrows heavily from the digital signal processing terminology, the last having its roots in the analogic world. Many algorithms are conceived in terms of filters, circuits and other similar legacy constructs, a fact that burdens the conceptual, theoretical framework of the area. Besides the introduction of a noveau envelope extraction method, we find that his formulation in terms of (differential) geometric concepts is, perhaps, as great a contribution as the algorithm per se.