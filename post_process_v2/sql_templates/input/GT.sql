CREATE TABLE `<table_name>`
(
    `CenterX` double  DEFAULT NULL,
    `CenterY` double  DEFAULT NULL,
    `Length`  double  DEFAULT NULL,
    `Width`   double  DEFAULT NULL,
    `ImageID` int(11) DEFAULT NULL,
    `LableID` int(11) DEFAULT NULL,
    `Area`    double  DEFAULT NULL,
    `BoxID`   int(11) DEFAULT NULL
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;