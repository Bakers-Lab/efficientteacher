CREATE TABLE `<table_name>`
(
    `阶段`        varchar(255) DEFAULT NULL,
    `缺陷类型`      varchar(255) DEFAULT NULL,
    `忽略缺陷`      varchar(255) DEFAULT NULL,
    `是否包含非重要缺陷` varchar(255) DEFAULT NULL,
    `GT`        int(11)      DEFAULT NULL,
    `pred标签`    int(11)      DEFAULT NULL,
    `pred框`     int(11)      DEFAULT NULL,
    `一致数量`      int(11)      DEFAULT NULL,
    `一致率`       varchar(255) DEFAULT NULL,
    `漏检数量`      int(11)      DEFAULT NULL,
    `漏检率`       varchar(255) DEFAULT NULL,
    `错检数量`      int(11)      DEFAULT NULL,
    `错检率`       varchar(255) DEFAULT NULL,
    `过杀数量`      int(11)      DEFAULT NULL,
    `过杀率`       varchar(255) DEFAULT NULL
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;