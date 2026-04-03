if (!require(ggplot2)) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
  library(ggplot2)
} else {
  library(ggplot2)
}

# Завантажуємо результати кластеризації, отримані в Python.
data <- read.csv("d:/курсач/banks_clustered.csv")

# Перетворюємо номер кластера на категоріальну змінну.
data$Cluster <- as.factor(data$Cluster)

# Будуємо більш інформативний графік:
# boxplot показує розподіл, а точки показують окремі банки.
plot <- ggplot(data, aes(x = Cluster, y = ROA, fill = Cluster)) +
  geom_boxplot(alpha = 0.6, outlier.color = "red", width = 0.55) +
  geom_jitter(width = 0.12, alpha = 0.75, size = 2, color = "black") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Розподіл рентабельності (ROA) за кластерами",
    x = "Номер кластера",
    y = "Значення ROA"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none"
  )

# Показуємо графік.
print(plot)

# Зберігаємо графік як окремий PNG-файл у папку проєкту.
ggsave(
  filename = "d:/курсач/cluster_boxplot.png",
  plot = plot,
  width = 9,
  height = 6,
  dpi = 300
)
