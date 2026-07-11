# Class Distribution

The compact five-class conversion produced 9,309 training objects and
1,106 validation objects.

| Class             | Training Objects | Training % | Validation Objects | Validation % |
| ----------------- | ---------------: | ---------: | -----------------: | -----------: |
| Vessel            |            6,356 |     68.28% |                820 |       74.14% |
| Buoy              |            1,566 |     16.82% |                125 |       11.30% |
| Swimmer           |              349 |      3.75% |                 26 |        2.35% |
| Paddle Board      |              154 |      1.65% |                 30 |        2.71% |
| Floating Obstacle |              884 |      9.50% |                105 |        9.49% |

The dataset exhibits substantial class imbalance. The training split
contains approximately 41 vessel annotations for every paddle-board
annotation. Vessel detections are therefore expected to dominate aggregate
metrics, while swimmer and paddle-board metrics may be less stable because
of their smaller sample sizes. The initial model will be trained without
custom balancing to establish a reproducible baseline. Later experiments
will evaluate targeted augmentation and higher input resolution for the
underrepresented classes.
