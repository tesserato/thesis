\section{Methodology}
Conceptually, one can think of the procedure here presented as a way to represent discrete, pitched, signal with a set of 3 tuples. Those tuples are composed, to describe them loosely, of a polynomial description of the envelope of the signal, that dictates the outer shape of the wave, or the volume of the sound as a function of time; another polynomial description, but this time for the (pseudo)-cycles of the wave, meant to represent the shape of each pseudo cycle of the signal, and a description of the beginning of each said pseudo cycle, and thus the end of the preceding one.

This aids in putting in perspective the broad strokes of the algorithm.

Using the algorithm presented in \textcite{2020TarjanoRobust} one can identify the frontiers, both positive and negative, of a digital signal: Those are, ideally, respectively composed of the maxima and minima of each pseudo-cycle of a discrete signal. 

If we disconsider inaccuracies we could use either one of the frontiers to infer the exact position, wavelength and amplitude of each pseudo-cycle and to, provided that the shape of the wave is also known, reconstruct the original wave with those pieces of information. Consider, for example, the digital signal presented in figure \ref{fig:signalenvelope}
and the extracted envelopes; the signal is an example of the voice of an alto singer, with its frontiers in black.

\begin{figure}[ht!]
  \centering
    %  \includegraphics[width=0.9\linewidth]{01signalenvelope}
     \includegraphics{01signalenvelope.pdf}
    % \def\svgwidth{\linewidth}
    %\input{01signalenvelope.pdf_tex}
  \caption{A discrete wave of the singing voice of an alto singer. The black lines are the frontiers; in the detail view, it can be seen that each region between two diamonds comprises a pseudo-cycle, as defined by the points belonging to the upper frontier. Similarly, two adjacent squares delimitate a pseudo-cycle, from the standpoint of the lower frontier.}
  \label{fig:signalenvelope}
\end{figure}

It can be seen that both frontiers provide approximately the same information about the location of each pseudo-cycle, with a $ \pi $ phase difference.

\begin{figure}[ht!]
  \centering
    %  \includegraphics[width=0.9\linewidth]{01signalenvelope}
     \includegraphics{02pcsraw-alto.pdf}
    % \def\svgwidth{\linewidth}
    %\input{01signalenvelope.pdf_tex}
  \caption{Raw pseudo-cycles superimposed as defined by the positive and negative frontiers, for the alto singer signal.}
  \label{fig:pcrawalto}
\end{figure}

We can normalize the amplitude the pseudo-cycles between zero and one, by dividing each of them by their maximum. The lengths of each pseudo-cycle can be normalized with the use of the Fourier transform; to avoid any loss of data, we can use the maximum wavelength as the standard.

\begin{figure}[ht!]
  \centering
    %  \includegraphics[width=0.9\linewidth]{01signalenvelope}
     \includegraphics{03pcs-alto.pdf}
    % \def\svgwidth{\linewidth}
    %\input{01signalenvelope.pdf_tex}
  \caption{Normalized pseudo-cycles superimposed as defined by the positive and negative frontiers, for an alto singer signal, and the average of all normalized pseudo-cycles.}
  \label{fig:pcalto}
\end{figure}

Comparing figures \ref{fig:pcrawalto} and \ref{fig:pcalto} we note the effect of the normalization, as a means to reduce the variance of the pseudo-cycles, specially in the regions near the discontinuities. This will enable a more accurate approximation of the average wave.


\begin{figure}[ht!]
  \centering
    %  \includegraphics[width=0.9\linewidth]{01signalenvelope}
     \includegraphics{04averagepc-alto.pdf}
    % \def\svgwidth{\linewidth}
    %\input{01signalenvelope.pdf_tex}
  \caption{Average pseudo-cycles, superimposed, after a change of phase of $-\pi$ on the average pseudo-cycle defined by the negative frontier, for an alto singer digital signal.}
  \label{fig:averagepc-alto}
\end{figure}

Figure \ref{fig:averagepc-alto} makes clear that the average pseudo-cycles for the alto singer signal, as estimated by both frontiers, are in reasonable agreement, despite the evident vertical asymmetries of the underlying wave.

That is not always the case, however, as figure \ref{fig:averagepc-piano33} illustrates. For a discrete signal representing the recording o the key 33 of a grand piano, we can see that the average pseudo-cycle inferred from the positive frontier is substantially different from the one obtained with the negative frontier.

The horizontal lines indicate the maximum of each pseudo cycle, being the point of adjacency between two subsequent pseudo cycles. For the average defined by the negative frontier we see that there is a local maxima near that middle that is almost as high as the global maxima, which is not the case as the average defined by the positive one.

\begin{figure}[ht!]
  \centering
    %  \includegraphics[width=0.9\linewidth]{01signalenvelope}
     \includegraphics{04averagepc-piano33.pdf}
    % \def\svgwidth{\linewidth}
    %\input{01signalenvelope.pdf_tex}
  \caption{Average pseudo-cycles, superimposed, after a change of phase of $-\pi$ on the average pseudo-cycle defined by the negative frontier, for a grand piano digital signal. The horizontal lines in the figure mark the points of maximum of each of the average pseudo cycles.}
  \label{fig:averagepc-piano33}
\end{figure}

We could take the average of those two representations, but that would dilute the parametric representation of the pseudo-cycle, besides introducing unnecessary complication in the stage of recreating the wave, as we would have to deal with redundant information about the location of each pseudo-cycle.

Instead, a measure of the statistical variance of the lengths of the pseudo cycles will be used in order to select the most appropriate frontier as guide for the construction of the average pseudo cycle waveform.

Figure \ref{fig:std-piano33} shows the distribution of the lengths of the pseudo cycles of the grand piano digital signal. We can use the standard deviation as this measure, and use the pseudo cycles as defined by the frontier with the least standard deviation.

\begin{figure}[ht!]
  \centering
    %  \includegraphics[width=0.9\linewidth]{01signalenvelope}
     \includegraphics{05std-piano33.pdf}
    % \def\svgwidth{\linewidth}
    %\input{01signalenvelope.pdf_tex}
  \caption{Histogram of the distribution of lengths of the pseudo cycles, as defined by the positive and negative frontiers, and the accompanying standard deviation, for a digital signal of the key 33 of a grand piano.}
  \label{fig:std-piano33}
\end{figure}

Thus, a tillable parametric representation of the average pseudo-cycle can be obtained via the least mean squares method; we are interested in the polynomial that best approximates the average pseudo cycle, respecting the conditions that the beginning and the end of this polynomial must coincide, as must the first derivative of that polynomial in those places, if we are to obtain a smooth transition between pseudo cycles.

From \textcite{2013SelesnickLeast} we know that a constrained least mean squares problem of the form $ \underset{A}{\min} ||Q A - Y||^2_2 \quad \text{subject to} \quad V A = B $ has a closed form approximate solution $ \hat{A} = \left(Q^{T} Q\right)^{-1} \left(Q^{T} Y - V^{T} \left(V \left(Q^{T} Q\right)^{-1} V^{T}\right)^{-1} V \left(Q^{T} Q\right)^{-1} Q^{T} Y - B \right) $, where $ Y $ is the vector we are interested in approximating.

We can proceed to define the real vector $ U = \{ u_0, u_1, \cdots, u_{m-1}\} $ as the average of all normalized pseudo-cycles, with $ m \in \mathbb{N} $ equal to the number of elements and the maximum length of the pseudo-cycles identified by the envelope detection algorithm.

As we are interested in maintaining $ C^0 $ and $ C^1 $ continuity between two adjacent pseudo-pulses, it is convenient to define the vector $ Y $ as two vectors $ E $ stacked, that is $ Y = \{ u_0, u_1, \cdots, u_{m-1}, u_0, u_1, \cdots, u_{m-1}\} $. 

$ A $, the vector of coefficients we are interested in estimating, will thus be composed of $ 2 (k + 1) $ items, where $ k $ is the order of the polynomial to be used in the approximation. Half those values are redundant, however, and $ A $ can thus be defined as $ A = \{ a_0, a_1, \cdots, a_k, a_0, a_1, \cdots, a_k \} $. $ Q $ is formed as usual, with the difference that
it is duplicated horizontally, as seen in equation \ref{eq:matrix}.

Anticipating that we will eventually evaluate this polynomial in various lengths, it is convenient to normalize $ X $ between $ 0 $ and $ 1 $, defining $ X = \left\{ \cancelto{0}{\frac{0}{m-1}}, \frac{1}{m-1}, \frac{2}{m-1}, \cdots, \cancelto{1}{\frac{m-1}{m-1}} \right\} $

\begin{equation} \label{eq:matrix}
\stackrel{Q_{m \times 2 (k + 1)}} {
\begin{bmatrix}
  1      & x_0     & \cdots & x_0^k   & 1      & x_0     & \cdots & x_0^k   \\
  1      & x_1     & \cdots & x_1^k   & 1      & x_1     & \cdots & x_1^k   \\
  \vdots & \vdots  & \ddots & \vdots  & \vdots & \vdots  & \ddots & \vdots  \\
  1      & x_{m-1} & \cdots & x_{m-1} & 1      & x_{m-1} & \cdots & x_{m-1} \\
\end{bmatrix}
}
\stackrel{A_{2 (k + 1) \times 1}}
{
\begin{bmatrix}
a_0    \\
a_1    \\
\vdots \\
a_k    \\
a_0    \\
a_1    \\
\vdots \\
a_k    \\
\end{bmatrix}
}
=
\stackrel{Y_{2 m \times 1}}
{
\begin{bmatrix}
y_0     \\
y_1     \\
\vdots  \\
y_{m-1} \\
y_0     \\
y_1     \\
\vdots  \\
y_{m-1} \\
\end{bmatrix}
}
\end{equation}

To maintain the continuity and smoothness, as well as the identity between the first and second half of the coefficients, $ V $ can be defined as in equation \ref{eq:constraint}.

\begin{equation} \label{eq:constraint}
\stackrel{V_{k + 3 \times 2 (k + 1)}} {
\begin{bmatrix}
  1      & 0       & \cdots & 0           & -1     & 0        & \cdots & 0                \\
  0      & 1       & \cdots & 0           & 0      & -1       & \cdots & 0                \\
  \vdots & \vdots  & \ddots & \vdots      & \vdots & \vdots   & \ddots & \vdots           \\
  0      & 0       & \cdots & 1           & 0      & 0        & \cdots & -1               \\
  1      & x_0     & \cdots & x_0^k       & -1     & -x_{m-1} & \cdots & -x_{m-1}^k       \\
  0      & 1       & \cdots & k x_0^{k-1} & 0      & -1       & \cdots & -k x_{m-1}^{k-1} \\
\end{bmatrix}
}
\stackrel{A_{2 (k + 1) \times 1}}
{
\begin{bmatrix}
a_0    \\
a_1    \\
\vdots \\
a_k    \\
a_0    \\
a_1    \\
\vdots \\
a_k    \\
\end{bmatrix}
}
=
\stackrel{B_{k + 2 \times 1}}
{
\begin{bmatrix}
0      \\
\vdots \\
0      \\
\end{bmatrix}
}
\end{equation}

The initial $k+1$ lines of $V$ assure that the first half of the coefficients of $A$ are equal to the second half, while the penultimate line ensures $C^0$ continuity and the last line ensures $C^1$ continuity, guaranteeing that the derivatives at the beginning and the end of the parametric pseudo cycles are equal.

\begin{figure}[ht!]
  \centering
    %  \includegraphics[width=0.9\linewidth]{01signalenvelope}
     \includegraphics{06approximation-alto}
    % \def\svgwidth{\linewidth}
    %\input{01signalenvelope.pdf_tex}
  \caption{Two adjacent pseudo-cycles. The detail view illustrates that $C^0$ and $C^1$ smoothness are maintained in the parametric representation when two reconstructions are stacked horizontally.}
  \label{fig:05approximation-alto}
\end{figure}