
# Upon submitting your revised manuscript, please upload the source files for your article. For additional details regarding acceptable file formats, please refer to the Guide for Authors at: 
Build a PDF of your manuscript source files on your computer and attach it with item type 'Manuscript'.
Bundle all manuscript source files in a single archive and attach it with item type 'LaTeX source files'. Source files include LaTeX files, BibTeX files, figures, tables, all LaTeX classes and packages that are not included in TeX Live and any other material that belongs to your manuscript.

When submitting your revised paper, we ask that you include the following items:

# Response to Reviewers (mandatory)

This should be a separate file labeled "Response to Reviewers" that carefully addresses, point-by-point, the issues raised in the comments appended below. You should also include a suitable rebuttal to any specific request for change that you have not made. Mention the page, paragraph, and line number of any revisions that are made.

# Manuscript and Figure Source Files (mandatory)

We cannot accommodate PDF manuscript files for production purposes. We also ask that when submitting your revision you follow the journal formatting guidelines. **Figures and tables may be embedded within the source file for the submission as long as they are of sufficient resolution for Production**. For any figure that cannot be embedded within the source file (such as *.PSD Photoshop files), the original figure needs to be uploaded separately. Refer to the Guide for Authors for additional information.

http://www.elsevier.com/journals/digital-signal-processing/1051-2004/guide-for-authors.

# Highlights (optional)

Highlights consist of a short collection of bullet points that convey the core findings of the article and should be submitted in a separate file in the online submission system. Please use 'Highlights' in the file name and include 3 to 5 bullet points (maximum 85 characters, including spaces, per bullet point). See the following website for more information

http://www.elsevier.com/highlights

# Graphical Abstract (optional).

Graphical Abstracts should summarize the contents of the article in a concise, pictorial form designed to capture the attention of a wide readership online. Refer to the following website for more information: http://www.elsevier.com/graphicalabstracts.


The revised version of your submission is due by May 30, 2021.

PLEASE NOTE: Digital Signal Processing would like to enrich online articles by displaying interactive figures that help the reader to visualize and explore your research results. For this purpose, we would like to invite you to upload figures in the MATLAB .FIG file format as supplementary material to our online submission system. Elsevier will generate interactive figures from these files and include them with the online article on SciVerse ScienceDirect. If you wish, you can submit .FIG files along with your revised submission.

Digital Signal Processing features the Interactive Plot Viewer, see: http://www.elsevier.com/interactiveplots. Interactive Plots provide easy access to the data behind plots. To include one with your article, please prepare a .csv file with your plot data and test it online at http://authortools.elsevier.com/interactiveplots/verification before submission as supplementary material.


Include interactive data visualizations in your publication and let your readers interact and engage more closely with your research. Follow the instructions here: https://www.elsevier.com/authors/author-services/data-visualization to find out about available data visualization options and how to include them with your article.




---------------------------

Intuitively, the temporal envelope can be understood as a slowly varying function that multiplies the signal, being responsible for its outer shape.

It is worth noting that, although that is the case throughout this work, the algorithm doesn't the original signal to be sampled at equal intervals.

Besides the most direct applications of this work to audio classification and synthesis, we foresee impact in compression techniques and machine learning approaches to audio. We briefly discuss some potential paths in this direction. The discrete curvature definition presented could also be extended to three-dimensional settings, to improve shape detection algorithms based on alpha shapes.

an intuitive way is to observe how well the carrier wave \textbf{c} conforms to the interval $ \{-1, 1\} $.

We begin this section suggesting an empirical metric, based on the behaviour of the carrier wave \textbf{c}, obtained dividing, element-wise, the original signal and the envelope obtained. We also comment on some metrics presented in the literature, and compare our algorithm with traditional demodulation approaches, both in relation to execution time and accuracy; to this last end, a numerical indicator is also introduced.

% We also illustrate how the method here presented can be adapted to isolate not only a single envelope of the wave, but the superior and inferior envelopes as well, that we call frontiers, 
% % TODO
% ending the section with a suggestion on how the proposed algorithm can be useful beyond envelope detection, by identifying the approximate locations of the pseudo-cycles in a quasi-periodic wave.

Our problem is to derive a measure of the average curvature of a discrete curve from its oscillations in the vertical axis.

that would , in the sense that pulse's amplitude $ \lvert w_j \lvert $, now mapped to the ordinates of a point, and a pulse's length, approximately equivalent to the horizontal distance between two consecutive points in the geometrical interpretation would define, on average, a square.

putting forth our own method, tailored to the definition of the equivalent circle of a wave; we end this section by showing how this circle can be used to identify the envelope of a discrete wave, in a similar approach to that of the alpha shapes theory, but taking advantage of some unique structure present in discrete waves.



We proceed to illustrate the representation of a discrete wave used in this work, largely based on the concept of dividing the wave into its constituent pulses, and how it can simplify the algorithm, reducing the high dimensionality generally present in digital representations of waves.

Next, we review methods for the estimation of the curvature of a discrete wave, after which we present our own; this method will then be used to estimate the equivalent circle, that will be used to define the envelope of the wave, via the procedure described in sequence.