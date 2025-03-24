DROP DATABASE IF exists`crypto`;
CREATE DATABASE `crypto`;
USE `crypto`;

CREATE TABLE `user` (
  `Id` int AUTO_INCREMENT,
  `Name` varchar(200) DEFAULT NULL,
  `Email` varchar(200) DEFAULT NULL,
  `Password` varchar(200) DEFAULT NULL,
  `Age` varchar(200) DEFAULT NULL,
  `Mob` varchar(200) DEFAULT NULL,
  PRIMARY KEY (`Id`)
);
