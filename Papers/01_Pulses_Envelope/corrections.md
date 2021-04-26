"Response to Reviewers"
Mention the page, paragraph, and line number of any revisions that are made.


# Reviewer #2: This is an interesting research. It seems that the authors have already put this article publicly accessible at arXiv.org. The authors are suggested to consider and address the following issues:

## Writing needs to be refined.
We extensively reviewed the text, rewriting many parts of it, besides reestructuring the sections to improve flow and coherence

## Issues with the Highlights: maximum 85 characters, including spaces, per bullet point

## Abstract looks too long. Should be presented in a more concise way.
We revised the abstract, removing superfluous information and verbose phrasing in order to make it more succinct, resulting in a reduction from the original 349 words to 248 words in the revised version.

## Citation of references should be in order and no jump of reference numbers.
Alphabetically ordering citations seems to be the default behaviour of biblatex, and we are sorry for the overlook. The issue was fixed by adding `sorting=none` to the biblatex import in the tex source.

## What's the Abscissa in the highlights (50k, 100k, etc.) and some figures?
The abscissas refer to the sample i of the digital signal. We revised the legends to make this information explicit, and made this more explicit in the paragraph where the discrete version of the problem is introduced.
<!-- TODO -->
## Legends in some figures are confusing and hard to distinguish, such as in Fig 2: signal and Hilbert Envelope.
We reworked all the figures, color coding the different entities shown when deemed helpful, with careful attention to the impact in the redability of the legends.

## Figure 3: the illustration of the carrier c and wave w does not comply with the common sense of envelope and modulation?
We changed the legend of the figure and the text of the revised version to make more explicit that the example was constructed from a known carrier and envelope to help in the visualization. 
<!-- TODO -->


## Definition of the envelope is very vague and controversial. What's the relationship with the conventional amplitude modulation?
We made more explicit that a general definition of envelope is still an open question in the literature, adding recent citations that corroborate this.

## For the proposed "Equivalent Circle Approach", an acceptable mathematical background and justification is missing.

## It's quite doubtful about the assumption of w = e ⊙ c?
We revised the equation to make clear that the relation holds by definition on our work.

## P8: the authors claimed that "the algorithm here presented satisfies the four conditions presented", however, no convincing evidence is provided. Please prove it mathematically.

## How was the benchmarking carried out? Details on the implementation should be provided.
We explained the benchmarking more extensively, citing that the implementations used were from the signal processing module of the Scipy Python library, and also made the source code for the tests, as well as the used samples, available at the repository dedicated for the work.


# Reviewer #4: This paper proposes a new method for envelope estimation by using the geometric properties of a discrete real signal. Using discrete curvatures to determine the envelope of a discrete signal is an interesting idea and results obtained by this proposed method seem to be acceptable. However, the reviewer cannot recommend the publication of the paper in its current form.The authors need to address the reviewers below and revise the paper accordingly:

## 1. The description of the key concepts is not clear. The paragraphs of the current paper were poorly written and readers cannot follow them easily. The authors are suggested to revise the paper extensively to improve its readability.
We revised the text extensively, with the aim of improving the overall readability and making the flow of ideas more linear. To that end, the overall structure of sections and subsections was substantially changed.

## 2. The authors describe basic concepts of convex and concave hulls in an intuitive manner and show that “the idea is to identify the local extrema that tough the envelope”. But the reviewer cannot fully understand the logical connection between them. The authors are suggested to explain their idea in a more explicit manner.


## 3. The reviewer cannot understand why filtering is needed in the pre-processing of Hilbert transform. Clarify the reasoning and necessity.
We explained in the text that the results of the pure Hilbert retain great part of the frequency content of the underlying wave, specially in the case of broadband signals, and added a figure to exemplify this effect in the case of a real world signal of an alto singer uttering a sustained note.

## 4. Envelopes obtained by the proposed method seem to be good, but those obtained by other comparable methods, such as Hilbert transform, are not reasonable. The reviewer believes that results obtained by Hilbert transform should not be bad as shown by the authors. 

We made more explicit that the envelopes obtained with the Hilbert transform, in the context of broadband signals, are generally considered suboptimal. We also included a comparison of the filtered and the unfiltered results for the sound of an alto singer.

### Besides, its end effects are serious. 

#### Is it caused by the filtering, which is mentioned by the reviewer in comment #3? 
The effects were caused by filtering the wave prior to the transform. We changed the order of the operations, eliminating the effects in the revised version.

#### Provide an explanation for the results in Figures 10, 11, and 12.



## 5. There are a lot of mistakes in the texts, please proofread the paper carefully. Some of them listed in the following:
We extensively revised the text.

### a. Full name at the first time for the abbreviate: being particularly illustrative of the potential synergy between geometric and DSP approaches.
We inserted the full name and abbreviation for DSP in its first use in the first paragraph of the revised version, and took steps to do the same with other abbreviations.

### b. Figure or table or equation should be illustrated:
#### From 3 and the discussion in the preceding chapter; 
#### The algorithm follows directly from the definition in 5 after noting; 
#### In 10 we illustrate the envelope extracted by the conventional algorithms;
#### The times taken for the algorithms compared here to process each wave are shown in 2;
#### Figure 14 shows the frequency-domain power spectrum for the wave and the carrier presented in 3…..

This issue arose from a misconception that those terms would be added during the latex compilation, discovered late in the process of elaborating the manuscript, that caused and many of the faulty references to remained unnoticed. We took steps to ensure proper indication of figures, tables and algorithms in the revised version.

### c. Wrong equation, should be 𝑟 : From figure 6 is easy to
  𝑘 = 𝑣𝑘,𝑥/ 𝑠𝑖𝑛(θ𝑘)
  see that 𝑟 .
  𝑘 = 𝑣𝑘,𝑥 𝑠𝑖𝑛(θ𝑘)

We thank the reviewer for pointing out this error. This mistake was fixed in the revised version. It turned out to be a typo, in the sense that the following equation, derived in part from this one, was correct. We seized the opportunity to also double check all other equations in the revised version. <!-- TODO -->

## 6. Figure 7 is shown in the manuscript, but there is no explanation on the main body for it. Please add an explanation in the revised paper.
We added an explanation to the figure 7 in the original paper, that illustrates the envelope of a guitar bend, as suggested.

# Reviewer #5: Once the carrier frequency is known the technique presented in this paper is a special kind of limited curvature interpolation technique.
We apologize if we misunderstood the commentary, but we think we addressed it when making clear that the wave in Fig. XX 
<!-- TODO -->
was constructed from a previously known carrier wave and envelope; that is the only instance when the carrier frequency, being a sinusoid, was known.


## The author should improve the citation by referencing corresponding techniques, such as K-curve, and bounded curvature interpolation methods. It is interesting to investigate the application of such technique to signal envelope extraction.




-----

Dear Mr. Tarjano,

Thank you for submitting your manuscript to Digital Signal Processing. Below you will find the reviewers' comments on your above-mentioned manuscript.  The reviewers have made suggestions which the Editor feels would improve your manuscript.  The Editor encourages you to consider these comments and make an appropriate revision of your manuscript.

Please submit your revision online within
May 30, 2021 by logging onto the Editorial Manager for Digital Signal Processing:
https://www.editorialmanager.com/dsp/
Your username is: tesserato
If you need to retrieve password details, please go to:
Can't remember your password?
To reset your password please try to sign in and click 'continue'. On the next screen click the 'forgot password' link and follow the steps to reset your password.    

The manuscript record can be found in the "Submissions Needing Revision" menu.

NOTE: Upon submitting your revised manuscript, please upload the source files for your article. For additional details regarding acceptable file formats, please refer to the Guide for Authors at: <<<enter link here>>>

When submitting your revised paper, we ask that you include the following items:

Response to Reviewers (mandatory)

This should be a separate file labeled "Response to Reviewers" that carefully addresses, point-by-point, the issues raised in the comments appended below. You should also include a suitable rebuttal to any specific request for change that you have not made. Mention the page, paragraph, and line number of any revisions that are made.

Manuscript and Figure Source Files (mandatory)

We cannot accommodate PDF manuscript files for production purposes. We also ask that when submitting your revision you follow the journal formatting guidelines.  Figures and tables may be embedded within the source file for the submission as long as they are of sufficient resolution for Production. For any figure that cannot be embedded within the source file (such as *.PSD Photoshop files), the original figure needs to be uploaded separately. Refer to the Guide for Authors for additional information.

http://www.elsevier.com/journals/digital-signal-processing/1051-2004/guide-for-authors.

Highlights (optional)

Highlights consist of a short collection of bullet points that convey the core findings of the article and should be submitted in a separate file in the online submission system. Please use 'Highlights' in the file name and include 3 to 5 bullet points (maximum 85 characters, including spaces, per bullet point). See the following website for more information

http://www.elsevier.com/highlights

Graphical Abstract (optional).

Graphical Abstracts should summarize the contents of the article in a concise, pictorial form designed to capture the attention of a wide readership online. Refer to the following website for more information: http://www.elsevier.com/graphicalabstracts.


The revised version of your submission is due by May 30, 2021.

PLEASE NOTE: Digital Signal Processing would like to enrich online articles by displaying interactive figures that help the reader to visualize and explore your research results. For this purpose, we would like to invite you to upload figures in the MATLAB .FIG file format as supplementary material to our online submission system. Elsevier will generate interactive figures from these files and include them with the online article on SciVerse ScienceDirect. If you wish, you can submit .FIG files along with your revised submission.

Digital Signal Processing features the Interactive Plot Viewer, see: http://www.elsevier.com/interactiveplots. Interactive Plots provide easy access to the data behind plots. To include one with your article, please prepare a .csv file with your plot data and test it online at http://authortools.elsevier.com/interactiveplots/verification before submission as supplementary material.


Include interactive data visualizations in your publication and let your readers interact and engage more closely with your research. Follow the instructions here: https://www.elsevier.com/authors/author-services/data-visualization to find out about available data visualization options and how to include them with your article.


Thank you and we look forward to receiving your revised manuscript.

With kind regards,

Ercan Engin Kuruoglu, Ph.D. Cantab
Editor-in-Chief
Digital Signal Processing
E-mail: dsp@elsevier.com


Note: While submitting the revised manuscript, please double check the author names provided in the submission so that authorship related changes are made in the revision stage. If your manuscript is accepted, any authorship change will involve approval from co-authors and respective editor handling the submission and this may cause a significant delay in publishing your manuscript.

Data in Brief (optional):
We invite you to convert your supplementary data (or a part of it) into an additional journal publication in Data in Brief, a multi-disciplinary open access journal. Data in Brief articles are a fantastic way to describe supplementary data and associated metadata, or full raw datasets deposited in an external repository, which are otherwise unnoticed. A Data in Brief article (which will be reviewed, formatted, indexed, and given a DOI) will make your data easier to find, reproduce, and cite.
 
You can submit to Data in Brief when you upload your revised manuscript. To do so, complete the template and follow the co-submission instructions found here: www.elsevier.com/dib-template. If your manuscript is accepted, your Data in Brief submission will automatically be transferred to Data in Brief for editorial review and publication.
 
Please note: an open access Article Publication Charge (APC) is payable by the author or research funder to cover the costs associated with publication in Data in Brief and ensure your data article is immediately and permanently free to access by all. For the current APC see: www.elsevier.com/journals/data-in-brief/2352-3409/open-access-journal
 
Please contact the Data in Brief editorial office at dib-me@elsevier.com or visit the Data in Brief homepage (www.journals.elsevier.com/data-in-brief/) if you have questions or need further information.




For further assistance, please visit our customer support site at http://help.elsevier.com/app/answers/list/p/7923. Here you can search for solutions on a range of topics, find answers to frequently asked questions and learn more about EM via interactive tutorials. You will also find our 24/7 support contact details should you need any further assistance from one of our customer support representatives.



---------------------------

Intuitively, the temporal envelope can be understood as a slowly varying function that multiplies the signal, being responsible for its outer shape.


Besides the most direct applications of this work to audio classification and synthesis, we foresee impact in compression techniques and machine learning approaches to audio. We briefly discuss some potential paths in this direction. The discrete curvature definition presented could also be extended to three-dimensional settings, to improve shape detection algorithms based on alpha shapes.