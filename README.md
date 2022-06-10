# skill_assessment_with_spatial_attention

**Code for EPIC@ICCV2019 paper**

<u>Manipulation-skill assessment from videos with spatial attention network</u>



Quatifying and ranking the action quality performed in videos by decaying the attention to the redundent background parts and modeling temporal relationships on key parts.

![framework](figs/Framework2.png)

We proposed a novel RNN-based spatial attention model that considers accumulated attention state from previous frames as well as high-level information about the progress of an undergoing task.


| ![Suturing_F5_Better](figs/Suturing_F5_Better.gif) | >    | ![Suturing_I5_Worse](figs/Suturing_I5_Worse.gif) |
| --- | ---- | --- |
| ![SonicDrawing_1_D5_Better](figs/SonicDrawing_1_D5_Better.gif) | >    | ![SonicDrawing_1_A1_Worse](figs/SonicDrawing_1_A1_Worse.gif) |

The approach is validated on four existing datasets of hand manipulation tasks, including Surgery, Drawing, Cooking and Using Chopsticks.