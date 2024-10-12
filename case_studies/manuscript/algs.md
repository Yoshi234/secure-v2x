# Algorithm Sketches

The algorithms need to be edited in several places: 

1. edit the secret share generation lines - make this more concise - 
   how can we shorten this bit to make a bit more clear
2. edit the secret reconstruction - say something like, $P_{1}$ sends his 
   shares to $P_{0}$ who reconstructs their shares 
3. create a dictionary object in the yolo algorithm which holds the 
   specific elements we use for later levels with the key they are 
   interested in and then just pull it if needed for the concatenation. 

## CryptoDrowsy

$$
\begin{align*}
& \textbf{Input: } P_{0} \text{ holds features } x\in\mathbb{R}^{1\times 384},\\
& \mskip{1em} P_{1} \text{ holds weights } W = \{W_{0}, W_{1}, W_{2}\} 
    \text{ and biases } b = \{b_{0}, b_{1}, b_{2}\} \\
& \textbf{Output: } P_{0} \text{ learns } c \in \{0,1\}, 
    \text{ the classification of its features} \\
& 1. \space P_{0} \text{ and } P_{1} \text{ invoke } F_{PRZS} 
    \text{ to generate secret shares of } x, \\
& \mskip{1em} \langle x\rangle_{0} \text{ and } \langle x\rangle_{1}, 
    \text{ and secret shares} \text{ of } W_{i},b_{i} \text{ for all } i \in \{0,1,2\}, \\
& \mskip{1em} \langle W_{i}\rangle_{0}, \langle b_{i}\rangle_{0} \text{ and }  
    \langle W_{i}\rangle_{1}, \langle b_{i}\rangle_{1} \\
& 2. \space P_{0} \text{ and } P_{1} \text{ invoke an instance of } F_{Conv2D}. 
    \text{ Each party}\text{ inputs its shares } \\
& \mskip{1em} \langle W_{0}\rangle_{k},\langle b_{0}\rangle_{k}, \langle x \rangle_{k} \text{ for }
    k\in \{0,1\} \text{ and }
    \text{obtains } \langle \mathcal{O}_{0}\rangle_{k} \\
& 3. \space \text{Each party calls } F_{BN} \text{ on its share } \\
& \mskip{1em } \langle \mathcal{O}_{0} \rangle_{k} \text{ for weights and biases } 
    \langle W_{1} \rangle_{k}, \langle b_{1} \rangle_{k} 
    \text{ to obtain } \langle \mathcal{O}_{1}\rangle_{k} \\
& 4. \space P_{0} \text{ and } P_{1} \text{ call } F_{ReLU} \text{ and input shares } 
    \langle \mathcal{O}_{1} \rangle_{k} \text{ and outputs } \langle \mathcal{O'}_{1} \rangle_{k} \\
& 5. \space P_{0} \text{ and } P_{1} \text{ reshape their shares } \langle \mathcal{O'}_{1} \rangle_{k} 
    \text{ to } 1 \times 32 \times 321 \text{ and locally } \\
& \mskip{1em} \text{ perform global pooling for each } 1 \times 321 \text{ feature map to output } \\
& \mskip{1em} \langle O''_{1} \rangle_{k} \text{ of shape } 1 \times 32 \\
& 6. \space P_{0} \text{ and } P_{1} \text{ input } \langle \mathcal{O''}_{1}\rangle_{k}, 
    \langle W_{2}\rangle_{k}, \langle b_{2}\rangle_{k} \text{ to } F_{FC} \text{ and output } 
    \langle \mathcal{O}_{2}\rangle_{k} \\
& 7. \space P_{1} \text{ sends its share } \langle \mathcal{O}_{2} \rangle_{1} \text{ to } P_{0} 
    \text{ who obtains } c
\end{align*}
$$

## FastSec-YOLO

$$
\begin{align*}
& \textbf{Input: } P_{0} \text{ holds normalized image } x\in\mathbb{R}^{k\times l\times 3} \\
& \mskip{1em} P_{1} \text{ holds weight sets } W = \{W_{0},W_{1},\dots,W_{n}\} \text{ and bias sets }
    \{b_{0}, b_{1},\dots, b_{n}\} \text{ for layers } \\
& \mskip{1em} L = (L_{0},L_{1},\dots,L_{n}) \\
& \textbf{Output: } P_{0} \text{ learns inference output } z \in \mathbb{R}^{m\times 85} \text{ where }
    m \text{ is the number raw of} \\ 
& \mskip{1em} \text{bounding boxes generated} \\
& 1. \space \text{Parties } P_{k},k\in\{0,1\} \text{ invoke } F_{PRZS} \text{ to secret shares of } x, 
    \langle x\rangle_{p}, \text{ and secret shares of } \\
& \mskip{1em} W_{i},b_{i} , \langle W_{i} \rangle_{k}, 
    \langle b_{i} \rangle_{k} \forall i \in \{0,1,\dots,n\} \\
& 2. \space \text{Let } a = 0 \\
& 2. \space \textbf{foreach } i \in \{0,1,\dots,n\} \textbf{ do} \\
& 3. \mskip{3em} \textbf{swtich } L_{i} \textbf{ do} \\
& 4. \mskip{6em} \textbf{case } \text{ConvBNSiLU}: \\
& 5. \mskip{9em} \langle \mathcal{O_{i}}\rangle_{k} \leftarrow F_{ConvBNSiLU}(
    \langle\mathcal{O_{i-1}}\rangle_{k},\langle W_{i}\rangle_{k},
    \langle b_{i} \rangle_{k}) \\
& 6. \mskip{6em} \textbf{case } \text{ConvBNSiLU} \space (a): \\
& 7. \mskip{9em}    \langle B_{a}\rangle_{k}=\langle\mathcal{O}_{i}\rangle_{k} \leftarrow 
    F_{ConvBNSiLU}(\langle\mathcal{O_{i-1}}\rangle_{k}, 
    \langle W_{i}\rangle_{k},\langle b_{i} \rangle_{k}) \\
& 8. \mskip{9em} a := a+1 \\
& 9. \mskip{6em} \textbf{case } \text{Conv2D } (a): \\
& 10. \mskip{8.6em} \langle\mathcal{B}_{a}\rangle_{k} \leftarrow F_{Conv2D}(
    \langle\mathcal{O}_{i-1}\rangle_{k},\langle W_{i} \rangle_{k}, \langle b_{i}\rangle_{k}) \\
& 8. \mskip{6em} \textbf{case } \text{C3}: \\
& 9. \mskip{9em} \langle \mathcal{O_{i}}\rangle_{k}\leftarrow 
    F_{C3}(\langle\mathcal{O_{i-1}}\rangle_{k}, 
    \langle W_{i}\rangle_{k},\langle b_{i} \rangle_{k}) \\
& 10. \mskip{5.6em} \textbf{case } \text{C3} \space (a): \\
& 11. \mskip{8.6em} \langle \mathcal{B}_{a}\rangle_{k}=\langle \mathcal{O_{i}}\rangle_{k}\leftarrow 
    F_{C3}(\langle\mathcal{O_{i-1}}\rangle_{k}, \langle W_{i}\rangle_{k},\langle b_{i} \rangle_{k}) \\
& 12. \mskip{8.6em} a:=a+1 \\
& 13. \mskip{5.6em} \textbf{case } \text{SPPF}: \\
& 14. \mskip{8.6em} \langle \mathcal{O}_{i}\rangle_{k} \leftarrow F_{SPPF}(
    \langle \mathcal{O}_{i-1}\rangle_{k} , \langle W_{i}\rangle_{k}, \langle b_{i}\rangle_{k})\\
& 14. \mskip{5.6em} \textbf{case } \text{Concat} \space (a): \\
& 15. \mskip{8.6em} \langle \mathcal{O}_{i} \rangle_{k} \leftarrow 
    F_{Concat}(\langle\mathcal{O_{i-1}}\rangle_{k}, \langle \mathcal{B}_{a}\rangle_{k}) \\
& 16. \mskip{8.6em} a := a - 1 \\
& 18. \mskip{5.6em} \textbf{case } \text{Upsample}:\\
& 19. \mskip{8.6em} \langle\mathcal{O_{i}}\rangle_{k}\leftarrow F_{Upsample}(
    \langle\mathcal{O}_{i-1}\rangle_{k}) \\
& 20. \space P_{1} \text{ sends its share } \langle \mathcal{B}_{0}\rangle_{1} \text{ to } 
    P_{0} \text{ who obtains } z = \sum_{k=1}^{1}\langle{B_{0}\rangle_{k}}\mod Q 
\end{align*}
$$

* What is the correct notation to use when writing that something copies or stores
  something else? We assume that $\langle \mathcal{O}_{i}\rangle_{k}$ is consumed and 
  transformed after passing through each layer, but we want to keep it or a copy of it for 
  concatenation at later layers (residual connections) in some cases. 
* is it better to explicilty specify the cases (like p1, p2, etc.) where we have concatenation from 
  an output in the backbone or head, or is it better to just have some variable which tracks which 
* how can we put together the output convolution layer? As I have it right now, there is an 
  issue with this algorithm where Conv2D uses the same weight/bias set as another ConvBNSiLU
  layer which is conducted at the same tree (hierarchy) level for $i$.

# Primitive Sub-Protocols

Don't bother with explaining the specifics of CrypTen-defined sub-protocols, simply 
explain what each of the high level (bundeled sub-protocols does later 
after the main body is written)

## CrypTenBN

$$
\begin{align*}

\end{align*}
$$

## CrypTReLU

# YOLO Sub-Protocols

## F(C3) Block

## F(ConvBNSiLU)

## F(SPPF)

## F(Bottleneck 1)

## F(Bottleneck 2)

## F(Concat)

## F(Upsample)

# Other

> computations that require conversions to binary secret shares